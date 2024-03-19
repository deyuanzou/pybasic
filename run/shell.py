from logo import *

import os
import basic

welcome()

while True:
    text = input('basic > ')
    if text.startswith('$'):
        if text[1:] == 'exit':
            print('bye')
            exit(0)
        else:
            os.system(text.replace('$', '', 1))
    elif text.strip() == '':
        continue
    else:
        result, error = basic.run("<stdin>", text)
        if error:
            print(error.as_string())
        elif result:
            if len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(result)

