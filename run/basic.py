##########################
# Specialness 作特殊处理的值
##########################
DIGITS = '0123456789'


#########################
# Error 错误
#########################
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}:{self.details}'
        result += f' in File {self.pos_start.fn}, Line {self.pos_start.row + 1}, Column {self.pos_start.col}'
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'IllegalCharError', details)


##########################
# Position 追踪输入文件的位置
##########################
class Position:
    def __init__(self, idx, row, col, fn, ft):  # fn为file name, ft为file text
        self.idx = idx
        self.row = row
        self.col = col
        self.fn = fn
        self.ft = ft

    def forward(self, current_char):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.idx += 1
            self.col += 0
        return self

    def copy(self):
        return Position(self.idx, self.row, self.col, self.fn, self.ft)


#########################
# Token 词法单元
#########################
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'


class Token:
    def __init__(self, type_, value=None):  # 这里使用type_是为了防止将内置名称type覆盖
        self.type = type_
        self.value = value

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'


#########################
#  Lexer 词法分析器
#########################
class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text  # text为输入文本
        self.pos = Position(-1, 0, -1, self.fn, self.text)  # pos为当前字符的位置
        self.current_char = None  # current_char为当前字符
        self.read_a_char()

    def read_a_char(self):
        self.pos.forward(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.read_a_char()
            elif self.current_char in DIGITS:
                tokens.append(self.make_num())
                self.read_a_char()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.read_a_char()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.read_a_char()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.read_a_char()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.read_a_char()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.read_a_char()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.read_a_char()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.read_a_char()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        return tokens, None

    def make_num(self):
        num_str = ''
        dot_count = 0

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.read_a_char()

        # 在make_num()的while循环作判断前向前读取了一个字符，
        # 回到make_tokens()的while循环时又向前读取了一个字符，
        # 因此需要回退一个字符。
        self.pos.idx -= 1

        if dot_count == 0:
            return Token(TT_INT, int(num_str))
        else:
            return Token(TT_FLOAT, float(num_str))


#########################
# Node 语法解析树的节点
#########################
class NumberNode:
    def __init__(self, tok):
        self.tok = tok

    def __repr__(self):
        return f'{self.tok}'


class BiOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


#########################
# Parser 语法解析器
#########################
class Parser:
    def __init__(self, tokens):
        self.current_tok = None
        self.tokens = tokens
        self.tok_idx = -1
        self.forward()

    def forward(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        return res

    def factor(self):
        tok = self.current_tok
        if tok.type in (TT_INT, TT_FLOAT):
            self.forward()
            return NumberNode(tok)

    def term(self):
        return self.bi_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.bi_op(self.term, (TT_PLUS, TT_MINUS))

    def bi_op(self, func, ops):
        left_node = func()
        while self.current_tok.type in ops:
            op_tok = self.current_tok
            self.forward()
            right_node = func()
            left_node = BiOpNode(left_node, op_tok, right_node)
        return left_node


#########################
# Run 主程序
#########################
def run(fn, text):
    # 生成Tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error
    # 生成语法树
    parser = Parser(tokens)
    ast = parser.parse()

    return ast, None
