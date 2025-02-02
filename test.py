import os
import requests

# Unsplash APIのアクセストークンをここに設定
API_KEY = 'I5wdrGyNp75qgYi6rawrgnSHRah11V8AJJ8RRa4-qNk'
BASE_URL = 'https://api.unsplash.com/photos/random'
QUERY = 'dog'  # 猫の画像を検索
NUMBER_OF_IMAGES = 150  # 取得する画像の枚数
DOWNLOAD_DIR = './dog_images'  # ダウンロード先のディレクトリ

# ダウンロード先ディレクトリを作成
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_images():
    params = {
        'query': QUERY,
        'count': NUMBER_OF_IMAGES,
        'client_id': API_KEY,
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        images = response.json()
        print(f"Found {len(images)} images.")
        
        for i, img in enumerate(images):
            img_url = img['urls']['regular']
            img_data = requests.get(img_url).content
            img_path = os.path.join(DOWNLOAD_DIR, f'dog_{i+117}.jpg')
            
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            print(f"Downloaded {i+1}/{NUMBER_OF_IMAGES}: {img_path}")
        
        print("Download complete.")
    else:
        print(f"Failed to fetch images. Status code: {response.status_code}")

if __name__ == '__main__':
    download_images()
