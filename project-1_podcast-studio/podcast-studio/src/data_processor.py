def process_data(source_type, content):
    if source_type == "text":
        if not content or len(content.strip()) == 0:
            return None
        return content.strip()
    return None
