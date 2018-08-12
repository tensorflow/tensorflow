def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
