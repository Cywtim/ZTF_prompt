import openai

client = openai.OpenAI(
    base_url="https://api.llm.ustc.edu.cn/v1", # 例如：https://api.openrouter.ai/api/v1
    api_key="sk-fBHDqvMbuPAf1EsENkqtsw"
)

response = client.chat.completions.create(
    model="qwen3.6-reasoner",  # 指定模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        #"url": "https://你的图片或Base64编码的图像地址"
                    },
                    #"message": {"hello"}
                },
                {"type": "text", "text": "请分析这张光变曲线图像"},
            ],
        }
    ]
)

print(response.choices[0].message.content)


"""
curl -X POST "https://api.llm.ustc.edu.cn/v1/chat/completions"
 -H "Content-Type: application/json" -H 
 "Authorization: Bearer sk-fBHDqvMbuPAf1EsENkqtsw" -d 
 '{"model": "qwen3.6-reasoner", "messages": [{"role": "user", "content": "hello"}]}'

"""