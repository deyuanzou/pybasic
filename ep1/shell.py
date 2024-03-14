import os

import basic

while True:
    text = input('basic > ')
    if text.startswith('$'):
        os.system(text.replace('$', '', 1))
    else:
        result, error = basic.run("<stdin>", text)
        if error:
            print(error.as_string())
        else:
            print(result)
