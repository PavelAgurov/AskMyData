def get_fixed_json(text : str) -> str:
    """
        Extract JSON from text
    """
    text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")
    
    # check if JSON is in code block
    if '```json' in text:
        open_bracket = text.find('```json')
        close_bracket = text.rfind('```')
        if open_bracket != -1 and close_bracket != -1:
            return text[open_bracket+7:close_bracket].strip()
    
    # check if JSON is in brackets
    tmp_text = text.replace("{", "[").replace("}", "]")
    open_bracket = tmp_text.find('[')
    if open_bracket == -1:
        return text
            
    close_bracket = tmp_text.rfind(']')
    if close_bracket == -1:
        return text
    
    return text[open_bracket:close_bracket+1]
