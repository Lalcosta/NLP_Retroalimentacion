# NLP_Retroalimentacion
_Author: Eduardo Acosta Hernández_

This is the repo for the NLP module

## Installation

1. Create a virtual environmet
2. Activate virtual environment
3. Clone this repository
4. Excecute `pip install -r requirements.txt`

## .env file
1. In order to run task 3 yo need ibm and microsoft keys
2. Create a .env file whith the following api keys

```
IBM_KEY= "<Your IBM key here>"
IBM_URL="<Your IBM url here>"

MICROSOFT_KEY="<Your MICROSOFT key here>"
MICROSOFT_REGION="<Your MICROSOFT region here>"
MICROSOFT_ENDPOINT="https://api.cognitive.microsofttranslator.com"
```
* NOTE: KEEP MICROSOFT ENDPOINT AS SHOWED
3. If you don't have ibm keys you can get started [here](https://cloud.ibm.com/catalog/services/language-translator?hideTours=true&=undefined)
4. If you don't have microsoft keys you can get started [here](https://learn.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=csharp)
5. Once you have created your .env file with the correct keys and references run in your terminal `python3 run.py`

![Task 3 graph](https://github.com/Lalcosta/NLP_Retroalimentacion/blob/main/Train%20and%20test%20errors.png)
