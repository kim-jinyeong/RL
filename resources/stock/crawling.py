import pandas as pd
import requests
from bs4 import BeautifulSoup


def print_stock_price(code, page_num):
    result = [[], [], [], [], [], []]
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    }
    for n in range(page_num):
        url = f'https://finance.naver.com/item/sise_day.nhn?code={code}&page={str(n + 1)}'
        r = requests.get(url, headers=headers)
        soup = None

        if r.status_code == 200:
            html = r.text
            soup = BeautifulSoup(html, 'html.parser')

        else:
            print(r.status_code)

        tr = soup.select('table > tr')

        for i in range(1, len(tr) - 1):
            if tr[i].select('td')[0].text.strip():
                result[0].append(tr[i].select('td')[0].text.strip())
                result[1].append(int(tr[i].select('td')[3].text.strip().replace(",", "")))
                result[2].append(int(tr[i].select('td')[4].text.strip().replace(",", "")))
                result[3].append(int(tr[i].select('td')[5].text.strip().replace(",", "")))
                result[4].append(int(tr[i].select('td')[1].text.strip().replace(",", "")))
                result[5].append(int(tr[i].select('td')[6].text.strip().replace(",", "")))

    df = pd.DataFrame(result)
    df = df.transpose()
    df.to_csv(f'./v1/{code}.csv', index=False)
if __name__ == '__main__':
    stock_code = '005930'
    pages = 75
    print_stock_price(stock_code, pages)
