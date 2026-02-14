import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

import yfinance as yf

# Load .env file for local development (optional, won't fail if not present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using system env vars (GitHub Actions)

# Configure logging (console only)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =============================================================================
# CONFIGURATION
# =============================================================================
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.05"))  # Default 5%
WARMUP_PERIOD_DAYS = 14  # Send daily emails for first 2 weeks

# =============================================================================
# ENVIRONMENT VARIABLES (from GitHub Secrets)
# =============================================================================
def get_env_variable(name: str, required: bool = True) -> str:
    """Get environment variable with error handling."""
    value = os.environ.get(name)
    if required and not value:
        raise ValueError(f"Required environment variable '{name}' is not set")
    return value or ""


def get_portfolio_holdings() -> dict:
    """
    Get portfolio holdings from environment variable.
    Expected format (JSON string):
    {"FXAIX": 100.5, "FZILX": 75.2, "FSSNX": 50.0, "QQQM": 25.3}
    """
    holdings_json = get_env_variable("PORTFOLIO_HOLDINGS")
    try:
        holdings = json.loads(holdings_json)
        logger.info(f"Loaded portfolio holdings: {list(holdings.keys())}")
        return holdings
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid PORTFOLIO_HOLDINGS JSON: {e}")


def get_target_allocations() -> dict:
    """
    Get target allocations from environment variable.
    Expected format (JSON string with percentages as decimals):
    {"FXAIX": 0.35, "FZILX": 0.25, "FSSNX": 0.25, "QQQM": 0.15}
    """
    allocations_json = get_env_variable("TARGET_ALLOCATIONS")
    try:
        allocations = json.loads(allocations_json)
        
        # Validate allocations sum to ~100%
        total = sum(allocations.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(f"Target allocations sum to {total*100:.1f}%, expected 100%")
        
        logger.info(f"Loaded target allocations: {allocations}")
        return allocations
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid TARGET_ALLOCATIONS JSON: {e}")


def get_start_date() -> datetime:
    """
    Get the start date for monitoring from environment variable.
    Expected format: YYYY-MM-DD (e.g., "2026-02-13")
    """
    start_date_str = get_env_variable("START_DATE")
    try:
        return datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid START_DATE format (expected YYYY-MM-DD): {e}")


def is_warmup_period() -> bool:
    """Check if we're still in the warmup period (first 2 weeks)."""
    start_date = get_start_date()
    days_elapsed = (datetime.now() - start_date).days
    in_warmup = days_elapsed < WARMUP_PERIOD_DAYS
    logger.info(f"Days since start: {days_elapsed}, Warmup period: {in_warmup}")
    return in_warmup


# =============================================================================
# PORTFOLIO ANALYSIS
# =============================================================================
def fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices for all tickers using yfinance batch download."""
    import requests as req
    import time
    
    prices = {}
    
    # First try: yfinance batch download
    try:
        logger.info(f"Attempting batch download for: {', '.join(tickers)}")
        data = yf.download(tickers, period="5d", progress=False, ignore_tz=True)
        
        if not data.empty:
            if len(tickers) == 1:
                if not data['Close'].empty:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
            else:
                for ticker in tickers:
                    try:
                        if ticker in data['Close'].columns:
                            close_prices = data['Close'][ticker].dropna()
                            if not close_prices.empty:
                                prices[ticker] = float(close_prices.iloc[-1])
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"Batch download failed: {e}")
    
    # Second try: Individual ticker lookup via Yahoo Finance API directly
    missing_tickers = [t for t in tickers if t not in prices]
    
    for ticker in missing_tickers:
        try:
            time.sleep(0.5)  # Small delay between requests
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = req.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('chart', {}).get('result', [])
                if result:
                    meta = result[0].get('meta', {})
                    price = meta.get('regularMarketPrice') or meta.get('previousClose')
                    if price:
                        prices[ticker] = float(price)
        except Exception as e:
            logger.warning(f"Direct API failed for {ticker}: {e}")
    
    # Log results
    for ticker, price in prices.items():
        logger.info(f"{ticker}: ${price:.2f}")
    
    missing = set(tickers) - set(prices.keys())
    for ticker in missing:
        logger.warning(f"No price data for {ticker}")
    
    return prices


def calculate_portfolio_value(holdings: dict, prices: dict) -> tuple:
    """
    Calculate total portfolio value and individual position values.
    Returns: (total_value, position_values)
    """
    position_values = {}
    for ticker, shares in holdings.items():
        if ticker in prices:
            position_values[ticker] = shares * prices[ticker]
    
    total_value = sum(position_values.values())
    return total_value, position_values


def calculate_current_allocations(position_values: dict, total_value: float) -> dict:
    """Calculate the current allocation percentages."""
    if total_value == 0:
        return {ticker: 0.0 for ticker in position_values}
    
    return {ticker: value / total_value for ticker, value in position_values.items()}


def check_drift(current_allocations: dict, target_allocations: dict, threshold: float) -> list:
    """
    Check if any position has drifted beyond the threshold.
    Returns a list of drift alerts.
    """
    alerts = []
    
    for ticker, target in target_allocations.items():
        current = current_allocations.get(ticker, 0.0)
        drift = current - target
        drift_pct = drift * 100
        
        if abs(drift) > threshold:
            direction = "OVER" if drift > 0 else "UNDER"
            alerts.append({
                "ticker": ticker,
                "target": target * 100,
                "current": current * 100,
                "drift": drift_pct,
                "direction": direction,
            })
            logger.warning(
                f"DRIFT ALERT: {ticker} is {direction}weight - "
                f"Target: {target*100:.1f}%, Current: {current*100:.1f}%, "
                f"Drift: {drift_pct:+.1f}%"
            )
    
    return alerts


# =============================================================================
# EMAIL NOTIFICATIONS
# =============================================================================
def send_email(alerts: list, portfolio_summary: dict, is_daily_summary: bool = False):
    """
    Send email notification about portfolio status.
    
    Args:
        alerts: List of drift alerts (may be empty)
        portfolio_summary: Full portfolio data
        is_daily_summary: If True, send regardless of alerts (warmup period)
    """
    email_address = get_env_variable("EMAIL_ADDRESS")
    email_password = get_env_variable("EMAIL_PASSWORD")
    recipient_email = get_env_variable("RECIPIENT_EMAIL", required=False) or email_address
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    has_drift = len(alerts) > 0
    
    # Build subject line based on status
    if has_drift:
        subject = f"🚨 REBALANCE NEEDED - IRA Portfolio Drift Alert - {date_str}"
    else:
        subject = f"✅ ALL GOOD - IRA Portfolio Status - {date_str}"
    
    # Build email content
    if has_drift:
        status_header = "Portfolio Drift Alert"
        status_message = f"Your IRA portfolio has drifted beyond the {DRIFT_THRESHOLD*100:.0f}% threshold."
        status_color = "#ff6b6b"
    else:
        status_header = "Daily Portfolio Status"
        status_message = f"Your IRA portfolio is balanced within the {DRIFT_THRESHOLD*100:.0f}% threshold. No action needed."
        status_color = "#51cf66"
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
    <h2 style="color: {status_color};">{status_header}</h2>
    <p>{status_message}</p>
    """
    
    # Show drift alerts if any
    if has_drift:
        body += f"""
    <h3>🔴 Drift Alerts</h3>
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Fund</th>
            <th>Target</th>
            <th>Current</th>
            <th>Drift</th>
            <th>Action</th>
        </tr>
    """
        for alert in alerts:
            action = "Consider selling" if alert["direction"] == "OVER" else "Consider buying"
            color = "#ffcccc" if alert["direction"] == "OVER" else "#ccffcc"
            body += f"""
        <tr style="background-color: {color};">
            <td><strong>{alert['ticker']}</strong></td>
            <td>{alert['target']:.1f}%</td>
            <td>{alert['current']:.1f}%</td>
            <td>{alert['drift']:+.1f}%</td>
            <td>{action}</td>
        </tr>
        """
        body += """
    </table>
    """
    
    # Always show full portfolio summary
    body += """
    <h3>📊 Full Portfolio Summary</h3>
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Fund</th>
            <th>Shares</th>
            <th>Price</th>
            <th>Value</th>
            <th>Target %</th>
            <th>Current %</th>
            <th>Drift</th>
        </tr>
    """
    
    for ticker, data in portfolio_summary["positions"].items():
        drift = data['current'] - data['target']
        drift_pct = drift * 100
        drift_str = f"{drift_pct:+.1f}%"
        
        # Color code based on drift severity
        if abs(drift) > DRIFT_THRESHOLD:
            row_color = "#ffcccc" if drift > 0 else "#ccffcc"
        else:
            row_color = "#ffffff"
        
        body += f"""
        <tr style="background-color: {row_color};">
            <td><strong>{ticker}</strong></td>
            <td>{data['shares']:.3f}</td>
            <td>${data['price']:.2f}</td>
            <td>${data['value']:,.2f}</td>
            <td>{data['target']*100:.1f}%</td>
            <td>{data['current']*100:.1f}%</td>
            <td>{drift_str}</td>
        </tr>
        """
    
    body += f"""
    </table>
    
    <p><strong>Total Portfolio Value:</strong> ${portfolio_summary['total_value']:,.2f}</p>
    <p style="color: #666; font-size: 12px;">
        This email was generated automatically by your IRA Portfolio Drift Checker.
    </p>
    </body>
    </html>
    """
    
    # Create and send email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_address
    msg["To"] = recipient_email
    msg.attach(MIMEText(body, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_address, email_password)
            server.sendmail(email_address, recipient_email, msg.as_string())
        logger.info(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Portfolio Drift Check")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Drift Threshold: {DRIFT_THRESHOLD*100:.1f}%")
    logger.info("=" * 60)
    
    try:
        # Load configuration from secrets
        holdings = get_portfolio_holdings()
        target_allocations = get_target_allocations()
        warmup = is_warmup_period()
        
        # Fetch current prices
        logger.info("Fetching current prices...")
        prices = fetch_current_prices(list(holdings.keys()))
        
        if not prices:
            raise ValueError("Could not fetch any prices")
        
        # Calculate portfolio values
        total_value, position_values = calculate_portfolio_value(holdings, prices)
        logger.info(f"Total Portfolio Value: ${total_value:,.2f}")
        
        # Calculate current allocations
        current_allocations = calculate_current_allocations(position_values, total_value)
        
        # Log current state
        logger.info("-" * 40)
        logger.info("Current Allocations:")
        for ticker in target_allocations:
            target = target_allocations[ticker] * 100
            current = current_allocations.get(ticker, 0) * 100
            logger.info(f"  {ticker}: {current:.1f}% (target: {target:.1f}%)")
        logger.info("-" * 40)
        
        # Check for drift
        alerts = check_drift(current_allocations, target_allocations, DRIFT_THRESHOLD)
        
        # Build portfolio summary for email
        portfolio_summary = {
            "total_value": total_value,
            "positions": {}
        }
        for ticker in target_allocations:
            portfolio_summary["positions"][ticker] = {
                "shares": holdings.get(ticker, 0),
                "price": prices.get(ticker, 0),
                "value": position_values.get(ticker, 0),
                "target": target_allocations[ticker],
                "current": current_allocations.get(ticker, 0),
            }
        
        # Decide whether to send email
        if warmup:
            # During warmup period: always send daily email
            logger.info("Warmup period active - sending daily summary email...")
            send_email(alerts, portfolio_summary, is_daily_summary=True)
        elif alerts:
            # After warmup: only send if there's drift
            logger.info(f"Found {len(alerts)} drift alert(s), sending notification...")
            send_email(alerts, portfolio_summary, is_daily_summary=False)
        else:
            logger.info("✅ No drift detected. Portfolio is balanced within threshold. (No email sent)")
        
        logger.info("Portfolio drift check completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during portfolio check: {e}")
        raise