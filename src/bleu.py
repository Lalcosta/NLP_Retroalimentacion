from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests, uuid, json
from nltk.translate.bleu_score import sentence_bleu
import os
from dotenv import load_dotenv
from statistics import mean


class Bleu_Score:
    
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
    
    def test(self):
        if len(self.text1)!=0:
            return "Bleu data is loaded"
        else: return " Bleu data not loaded"
        
    def limit_hundred_data(self):
        hundred_text1=[]
        hundred_text2=[]
        with open(self.text1, "r", encoding='utf-8') as one, open(self.text2,"r",encoding='utf-8') as two:
            for i in range(100):
                hundred_text1.append(one.readline()[:-1])
                hundred_text2.append(two.readline()[:-1])
        
        self.hundred_text1=hundred_text1
        self.hundred_text2=hundred_text2
        
        
    
    def bleu(self):
        load_dotenv()
        
        IBM_KEY=os.getenv('IBM_KEY')
        IBM_URL=os.getenv('IBM_URL')
        
        ibm_authenticator= IAMAuthenticator(IBM_KEY)
        ibm_translator=LanguageTranslatorV3(version="2018-05-01", authenticator=ibm_authenticator)
        ibm_translator.set_service_url(IBM_URL)
        
        MICROSOFT_KEY=os.getenv('MICROSOFT_KEY')
        MICROSOFT_REGION=os.getenv('MICROSOFT_REGION')
        MICROSOFT_ENDPOINT=os.getenv('MICROSOFT_ENDPOINT')
        
        microsoft_constructed_url = MICROSOFT_ENDPOINT+ "/translate"
        microsoft_params={
            'api-version': '3.0',
            'from': 'en',
            'to': 'es'
        }
        microsoft_headers = {
            'Ocp-Apim-Subscription-Key': MICROSOFT_KEY,
            'Ocp-Apim-Subscription-Region': MICROSOFT_REGION,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        microsoft_bleu_list=[]
        ibm_bleu_list=[]
        
        for i in range(len(self.hundred_text1)):
            microsoft_body=[{'text':self.hundred_text1[i]}]
            microsoft_request=requests.post(microsoft_constructed_url,params=microsoft_params,headers=microsoft_headers,json=microsoft_body)
            microsoft_response=microsoft_request.json()
            microsoft_translated= microsoft_response[0]["translations"][0]["text"]
            microsoft_bleu_list.append(sentence_bleu(self.hundred_text2[i].split(), microsoft_translated.split()))
            
            ibm_translation=ibm_translator.translate(text=self.hundred_text1[i],model_id="en-es").get_result()
            ibm_translated=ibm_translation["translations"][0]["translation"]
            ibm_bleu_list.append(sentence_bleu(self.hundred_text2[i].split(),ibm_translated.split()))
          
        print(f"MICROSOFT_TRANSLATOR: {mean(microsoft_bleu_list)}")
        print(f"IBM_TRANSLATOR: {mean(ibm_bleu_list)}")
        

