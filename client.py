# client.py
#Python调用 API 的客户端封装
import requests
import logging

class Phi3Client:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        logging.basicConfig(level=logging.INFO, format="[client] %(message)s")
        self.logger = logging.getLogger("Phi3Client")

    def predict(self, sentence: str) -> str:
        """单条预测"""
        url = f"{self.base_url}/predict"
        payload = {"sentence": sentence}

        self.logger.info(f"POST {url} | {sentence}")

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

        return response.json().get("category")


def main():
    client = Phi3Client()
    print("=== Phi3Client 交互式预测 ===")
    print("输入句子即可获得预测分类，输入 'exit' 退出。")

    while True:
        sentence = input("输入句子: ").strip()
        if sentence.lower() == "exit":
            print("退出预测。")
            break
        if not sentence:
            continue

        category = client.predict(sentence)
        print(f"预测类别: {category}\n")


if __name__ == "__main__":
    main()
