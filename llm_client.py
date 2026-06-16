import  os
from idlelib.rpc import response_queue

from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Dict


#加载.env文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为本书“Hello Agents”定制的客户端
    它用于调用任何兼容的OpenAI接口的服务，并默认使用流式响应
    """
    def __init__(self,model:str=None,apiKey:str=None,baseUrl:str=None,timeout:int=None):
        """
        初始化客户端
        :param model:模型
        :param apiKey: 秘钥
        :param baseUrl: 模型地址
        :param timeout: 超时设置
        """
        self.model=model or os.getenv("LLM_MODEL_ID")
        self.apiKey=apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl=baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout=timeout or int(os.getenv("TIMEOUT",60))

        if not all([self.model,self.apiKey,self.baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client=OpenAI(api_key=self.apiKey,base_url=self.baseUrl,timeout=self.timeout)
    def think(self,messages:List[Dict[str,str]],temperature:float=0)-> str:
        """
        调用大预言模型进行思考，并返回其响应
        :param messages: 用户消息，提示词
        :param temperature: 温度
        :return: 大模型返回结果
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response=self.client.chat.completions.create(
                    messages=messages,
                    model=self.model, # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
                    stream=True,
                    temperature=temperature
                )
            print("处理流式响应中...")
            collected_content=[]
            for chunk in response:
                if not chunk.choices:
                    continue
                content=chunk.choices[0].delta.content or ""
                print(content,end="",flush=True)
                collected_content.append(content)
            print()#在流式输出后换行
            return "".join(collected_content)
        except Exception as e:
            print(f"模型调用失败: {e}")
            return ""

#测试
if __name__ == '__main__':
    print(os.getenv("LLM_BASE_URL"))
    try:
        llmClient=HelloAgentsLLM()
        messages=[
            {"role":"system","content":"你是一个助手"},
            {"role":"user","content":"什么mcp协议？"},
        ]
        print("---调用LLM----")
        responseText=llmClient.think(messages)
        if responseText.strip():
            print("\n\n---模型完整响应---")
            print(responseText)
    except ValueError as e:
        print(e)
