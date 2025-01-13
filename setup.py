from playwright.sync_api import Playwright, sync_playwright
from urllib.parse import parse_qs,urlparse,quote
import pyotp
import requests
from config import *

auth_url = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apiKey}&redirect_uri={rurl}'

def getAccessToken(code):
    url = 'https://api.upstox.com/v2/login/authorization/token'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'code': code,
        'client_id': apiKey,
        'client_secret': secretKey,
        'redirect_uri': rurl,
        'grant_type': 'authorization_code',
    }
 
    response = requests.post(url, headers=headers, data=data)
    json_response = response.json()
    access_token = json_response['access_token']

    return access_token


def run(playwright: Playwright) -> str:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    with page.expect_request(f"*{rurl}/?code*") as request:
        page.goto(auth_url)
        page.locator("#mobileNum").click()
        page.locator("#mobileNum").fill(mobile_no)
        page.get_by_role("button", name="Get OTP").click()
        page.locator("#otpNum").click
        otp = pyotp.TOTP(totp_key).now()
        page.locator("#otpNum").fill(otp)
        page.get_by_role("button", name="Continue").click()
        page.get_by_label("Enter 6-digit PIN").fill(pin)
        page.get_by_role("button", name="Continue").click()
        page.wait_for_load_state()

    url = request.value.url 
    parsed = urlparse(url)
    code = parse_qs(parsed.query)['code'][0]
    context.close()
    browser.close()
    return code


with sync_playwright() as playwright:
    code = run(playwright)

access_token = getAccessToken(code)
#print('the access token is: ', access_token)