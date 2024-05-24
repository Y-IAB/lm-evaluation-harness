def translation_mc_get_prompt_ko_en(doc: dict) -> str:
    return f"""You are a helpful AI assistant that is translating Korean to English. The Korean text is given below. Please start the translated text in English.
    Korean: {doc["segments"]["sourceText"]}"""

def translation_mc_get_prompt_en_ko(doc: dict) -> str:
    return f"""You are a helpful AI assistant that is translating English to Korean. The English text is given below. Please start the translated text in Korean.
    English: {doc["segments"]["sourceText"]}"""

def translation_mc_get_target(doc: dict) -> str:
    return f"""{doc["segments"]["targetText"]}"""

def translation_mc_get_choices(doc: dict) -> str:
    return [f"""{doc["segments"]["mtText"]}""", f"""{doc["segments"]["targetText"]}"""]