from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests, uuid, json
from nltk.translate.bleu_score import sentence_bleu
import os
from dotenv import load_dotenv
from statistics import mean

#Class for determining bleu score in models
class Bleu_Score:
    
    #Init class with two datasets
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
    
    #Test to verify that datasets are stored
    def test(self):
        if len(self.text1)!=0:
            return "Bleu data is loaded"
        else: return " Bleu data not loaded"
    
    #Function to limit data to 100 lines lenght
    def limit_hundred_data(self):
        #Lists for the 100 lines datasets
        hundred_text1=[]
        hundred_text2=[]
        #Open datasets and append 100 first lines to lists
        with open(self.text1, "r", encoding='utf-8') as one, open(self.text2,"r",encoding='utf-8') as two:
            for i in range(100):
                #Append line deleting las character, thats because a "\n" was added to every line
                hundred_text1.append(one.readline()[:-1])
                hundred_text2.append(two.readline()[:-1])
        #Defining new class variables
        self.hundred_text1=hundred_text1
        self.hundred_text2=hundred_text2
        
        
    #Function to get bleu score
    def bleu(self):
        #Load .env file
        load_dotenv()
        #Get IBM KEYS
        IBM_KEY=os.getenv('IBM_KEY')
        IBM_URL=os.getenv('IBM_URL')
        #Authenticate
        ibm_authenticator= IAMAuthenticator(IBM_KEY)
        #Create a translator
        ibm_translator=LanguageTranslatorV3(version="2018-05-01", authenticator=ibm_authenticator)
        #Connecting to the service
        ibm_translator.set_service_url(IBM_URL)
        #Get Microsoft keys
        MICROSOFT_KEY=os.getenv('MICROSOFT_KEY')
        MICROSOFT_REGION=os.getenv('MICROSOFT_REGION')
        MICROSOFT_ENDPOINT=os.getenv('MICROSOFT_ENDPOINT')
        #Create a translator
        microsoft_constructed_url = MICROSOFT_ENDPOINT+ "/translate"
        #Defining translator parameters
        microsoft_params={
            'api-version': '3.0',
            'from': 'en',
            'to': 'es'
        }
        #Access to the translator
        microsoft_headers = {
            'Ocp-Apim-Subscription-Key': MICROSOFT_KEY,
            'Ocp-Apim-Subscription-Region': MICROSOFT_REGION,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        #Empty lists for the bleu scores
        microsoft_bleu_list=[]
        ibm_bleu_list=[]
        
        for i in range(len(self.hundred_text1)):
            #FOR MICROSOFT:
            #Create json [{text:line}]
            microsoft_body=[{'text':self.hundred_text1[i]}]
            #Send sentence to the translator
            microsoft_request=requests.post(microsoft_constructed_url,params=microsoft_params,headers=microsoft_headers,json=microsoft_body)
            #Get translator response
            microsoft_response=microsoft_request.json()
            #Access to the translated line
            microsoft_translated= microsoft_response[0]["translations"][0]["text"]
            #Get bleu score for the translated line comparing with hundred_text[i]
            microsoft_bleu_list.append(sentence_bleu(self.hundred_text2[i].split(), microsoft_translated.split()))
            
            #FOR IBM:
            #Translate line
            ibm_translation=ibm_translator.translate(text=self.hundred_text1[i],model_id="en-es").get_result()
            #Access to the translated line
            ibm_translated=ibm_translation["translations"][0]["translation"]
            #Get bleu score for the translated line comparing with hundred_text[i]
            ibm_bleu_list.append(sentence_bleu(self.hundred_text2[i].split(),ibm_translated.split()))
            
        #Print bleu scores
        print(f"MICROSOFT_TRANSLATOR: {mean(microsoft_bleu_list)}")
        print(f"IBM_TRANSLATOR: {mean(ibm_bleu_list)}")
        

