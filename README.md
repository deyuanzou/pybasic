# 项目介绍
用python实现的basic解释器



# 运行方式

- 点击`run`目录下的`run.bat`或点击`debug`目录下的`debug.bat`

![](.\needme\Snipaste_2024-03-24_00-55-11.png)

![](.\needme\Snipaste_2024-03-24_00-56-00.png)

- 也可以在命令行中输入`run.bat`或`debug.bat`

![](.\needme\Snipaste_2024-03-24_01-14-20.png)

![](.\needme\Snipaste_2024-03-24_01-15-46.png)



# 目录结构

目录结构如下：

![](.\needme\Snipaste_2024-03-24_00-45-49.png)

如上所示，`debug`目录和`run`目录的所有文件名除了.bat文件外是一样的，两者的区别如下：

- 运行`debug`目录下的`debug.bat`, 会输出解释器的整个工作流程，即：

  `词法分析`→`语法分析`→`求值计算`→`打印显示`

- 运行`run`目录下的`run.bat`, 会直接得到执行结果

比如，在解释器中输入5*2+3

- `debug.bat`输出如下：

![](.\needme\Snipaste_2024-03-24_01-14-40.png)

- `run.bat`输出如下：

![](.\needme\Snipaste_2024-03-24_01-16-45.png)



# 语法

## 命令

- 在终端中输入以`$`开头的文本，解释器将执行对应的命令

- 比如，清屏输入`$cls`，退出程序输入`$exit`

![](.\needme\Snipaste_2024-03-24_01-23-51.png)

## 变量

- 定义变量使用`var`关键字，变量的命名规则与C语言相同

- 变量的类型有`数值`、`列表`、`布尔`、`字符串`，其中`布尔`的定义与C语言相同，即0值为`false`、非0值为`true`

- 变量声明时无需指定类型，解释器会进行**自动推导**

- 变量支持`+`、`-`、`*`、`/`和`^`，其中`^`为幂运算符

![](.\needme\Snipaste_2024-03-24_01-52-22.png)

- 需要注意的是，不同于一般的编程语言，你无法对一个变量进行重新赋值，但你可以重新声明将其覆盖

![](.\needme\Snipaste_2024-03-24_01-53-11.png)

- 声明`字符串`时用一对`"`包裹，不支持`'`

![](.\needme\截图20240324102405.png)

- 声明`列表`时用`[]`包裹，**与python类似**，`列表`里每个元素的数据类型可以不同，同时`列表`可以嵌套，**需要注意**的是：一旦对`列表`使用了`运算符`，就会改变`列表`本身元素的值

![](.\needme\截图20240324104209.png)

- 另外一点值得注意的是，我已经预定义了4个变量，这意味着你在命名变量时应该**避免与它们冲突**，同时也意味着你可以不声明直接使用它们

![](.\needme\截图20240324105455.png)

## 函数

- 定义函数使用`fun`关键字，函数的命名规则与变量的命名规则相同
- 函数会返回最后一个表达式的值

![](.\needme\截图20240324110923.png)

- 支持匿名函数，同时可以将函数作为变量赋值，这意味着你可以进行函数式编程

![](.\needme\截图20240324112518.png)

- 可以在函数内部定义函数，并可以通过`return`语句将其返回

![](.\needme\Snipaste_2024-03-24_16-09-43.png)

- 下面是两个内置函数，其中`print`用于打印输出，`input`用于接收输入

![](.\needme\Snipaste_2024-03-24_16-36-24.png)

## 语句

- 与条件语句相关的关键字有`if`、`then`、`elif`、`else`，条件语句也会有返回值

![](.\needme\Snipaste_2024-03-24_13-16-49.png)

- 与循环语句相关的关键字有`for`、`to`、`step`、`while`、`then`，同样地，它们也会有返回值，但不同于一般编程语言的是我们将返回一个`列表`，`列表`的每个元素分别与每次循环的结果一一对应

![](.\needme\Snipaste_2024-03-24_15-29-25.png)

![](.\needme\Snipaste_2024-03-24_15-30-47.png)

- 与控制语句相关的关键字有`continue`、`break`、`return`，这些语句的用法与C语言相同，因此不再赘述，除此之外`return`的特殊用法已经在函数中有过说明
- 多行语句用`;`或`\n`进行分隔

## 注释

- 目前只支持单行注释，在当前行`#`后的文本将被视为注释

## 文件

- 支持以读取文本文件的方式执行代码
- 源代码文件只要是文本文件即可，对后缀名无要求，但建议以`.b`为后缀名
- 缩进与空白对源文件没有影响，但建议采取适当的缩进以使得代码更加易于阅读

![](.\needme\Snipaste_2024-03-24_16-30-40.png)

![](.\needme\Snipaste_2024-03-24_16-31-23.png)



# 我的github

[deyuanzou/pybasic: 用python实现的basic解释器 (github.com)](https://github.com/deyuanzou/pybasic)



# 参考
https://github.com/davidcallanan/py-myopl-code