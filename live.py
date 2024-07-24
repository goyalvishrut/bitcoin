from bs4 import BeautifulSoup
import requests
import time
import streamlit as st

def getprice(coin):
    url = 'https://www.google.com/search?q='+coin+'+price+live+in+inr'

    HTML =requests.get(url)

    soup = BeautifulSoup(HTML.text, 'html.parser')
    
    text = soup.find('div', attrs={'class':'BNeawe iBp4i AP7Wnd'}).find('div', attrs={'class':'BNeawe iBp4i AP7Wnd'}).text
    
    return text

def main():
    p=[]
    ls=-1
    while True:
        c='ethereum'
        price=getprice(c)
        if price!=ls:
            p.append(price)
            st.write(p[-1])
            ls=price

main()