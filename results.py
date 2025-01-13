import re
from playwright.sync_api import Playwright, sync_playwright, expect
import pandas as pd

df = pd.read_csv('stock_contracts.csv')
#df = df.iloc[204:]
stock_list = ['AAREY DRUGS & PHARM LTD','AARTI INDUSTRIES LTD','ABB INDIA LIMITED']

#stock_list = df['name'].tolist()
#print(stock_list)

links=[]

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    for name in stock_list:
        page = context.new_page()
        page.goto("https://economictimes.indiatimes.com/")
        page.goto("https://economictimes.indiatimes.com/defaultinterstitial.cms")
        page.get_by_text("Click here to go to").click()
        page.get_by_placeholder("Search Stock Quotes, News,").click()
        page.get_by_placeholder("Search Stock Quotes, News,").fill(name)
        page.get_by_placeholder("Search Stock Quotes, News,").press("Enter")
        with page.expect_popup() as page1_info:
            page.get_by_role("link", name=f"View all {name} Share News &").click()
        link = page1_info.value
        links.append(link.url)

    context.close()
    browser.close()
        #return links

with sync_playwright() as playwright:
    run(playwright)

print(links)

