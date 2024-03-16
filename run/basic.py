#########################
# Import 项目依赖
#########################
from string_with_arrows import *

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
        result += '\n\n' + string_with_arrows(self.pos_start.ft, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'IllegalCharError', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'InvalidSyntaxError', details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}:{self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ft, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result += f'File {pos.fn}, Line {str(pos.row + 1)}, Col {str(pos.col)} in {ctx.display_name}\n'
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result


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

    def forward(self, current_char=None):
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
TT_POW = 'POW'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):  # 这里使用type_是为了防止将内置名称type覆盖
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.forward()

        if pos_end:
            self.pos_end = pos_end

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
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.read_a_char()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.read_a_char()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.read_a_char()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_num(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

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
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)


#########################
# Node 语法解析树的节点
#########################
class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class BiOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'


#########################
# ParseResult 解析结果
#########################
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node
        return res

    def success(self, node):
        self.node = node
        return self

    def fail(self, error):
        self.error = error
        return self


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
        if not res.error and self.current_tok.type != TT_EOF:
            return res.fail(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.forward())
            return res.success(NumberNode(tok))
        elif tok.type == TT_LPAREN:
            res.register(self.forward())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.forward())
                return res.success(expr)
            else:
                return res.fail(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))

        return res.fail(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected int, float, '+', '-' or ')'"
        ))

    def power(self):
        return self.bi_op(self.atom, (TT_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok
        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.forward())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def term(self):
        return self.bi_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.bi_op(self.term, (TT_PLUS, TT_MINUS))

    def bi_op(self, func_a, ops, func_b=None):
        if func_b is None:
            func_b = func_a

        res = ParseResult()
        left_node = res.register(func_a())
        if res.error:
            return res
        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.forward())
            right_node = res.register(func_b())
            if res.error:
                return res
            left_node = BiOpNode(left_node, op_tok, right_node)
        return res.success(left_node)


#########################
# RTResult 运行结果
#########################
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def fail(self, error):
        self.error = error
        return self


#########################
# Value 表达式的值
#########################
class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start,
                    other.pos_end,
                    'Divided by zero',
                    self.context
                )
            return Number(self.value / other.value).set_context(self.context), None

    def powed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def __repr__(self):
        return str(self.value)


#########################
# Context 上下文信息
#########################
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos


#########################
# Interpreter 解释器
#########################
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_BiOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        if node.op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_to(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powed_to(right)

        if error:
            return res.fail(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_to(Number(-1))

        if error:
            return res.fail(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


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
    if ast.error:
        return None, ast.error
    # 运行程序
    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
