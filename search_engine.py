# Copyright (c) 2026 tuanzi. All rights reserved.
import re
import shlex
import pandas as pd

class BooleanSearchEngine:
    """
    布尔搜索引擎，支持 AND, OR, NOT 操作符以及括号优先级。
    """
    def __init__(self, df, scope="标题 + 摘要"):
        """
        初始化搜索引擎。
        
        Args:
            df (pd.DataFrame): 包含论文数据的 DataFrame。
            scope (str): 搜索范围，"仅标题" 或 "标题 + 摘要"。默认为 "标题 + 摘要"。
        """
        self.df = df
        self.scope = scope

    def search(self, query):
        """
        执行搜索的主入口。
        
        Args:
            query (str): 用户输入的查询字符串，支持布尔逻辑 (e.g., "A AND (B OR C)")。
            
        Returns:
            pd.DataFrame: 匹配查询条件的论文数据。
        """
        if not query or not query.strip():
            return pd.DataFrame()
            
        try:
            # 1. 预处理查询字符串
            tokens = self._tokenize(query)
            if not tokens:
                return pd.DataFrame()
                
            # 2. 将中缀表达式转换为后缀表达式 (Shunting Yard Algorithm)
            postfix = self._shunting_yard(tokens)
            
            # 3. 评估后缀表达式
            mask = self._evaluate_postfix(postfix)
            
            return self.df[mask]
        except Exception as e:
            # 如果解析失败，回退到简单的全字匹配 AND 搜索
            print(f"Search parsing error: {e}")
            return self._fallback_search(query)

    def _tokenize(self, query):
        """
        将查询字符串分解为 token。
        支持的操作符: AND, OR, NOT, (, )
        支持短语: "phrase search"
        
        Args:
            query (str): 原始查询字符串。
            
        Returns:
            list: Token 列表。
        """
        # 预处理：将中文括号转为英文，确保操作符大写
        query = query.replace('（', '(').replace('）', ')')
        
        # 使用 shlex 处理引号内的短语
        try:
            lexer = shlex.shlex(query, posix=True)
            lexer.whitespace_split = False
            lexer.wordchars += '*._-' # 允许这些字符作为词的一部分
            
            tokens = []
            for token in lexer:
                if token in ['(', ')']:
                    tokens.append(token)
                elif token.upper() in ['AND', 'OR', 'NOT']:
                    tokens.append(token.upper())
                else:
                    tokens.append(token)
            
            # 插入默认的 AND 操作符
            # 例如: "embedding ai" -> "embedding AND ai"
            # 规则: 如果两个操作数相邻，或者 ) 和操作数相邻，插入 AND
            final_tokens = []
            operators = {'AND', 'OR', 'NOT', '('}
            
            for i, token in enumerate(tokens):
                final_tokens.append(token)
                
                if i < len(tokens) - 1:
                    current_is_operand = token not in operators and token != ')'
                    next_is_operand = tokens[i+1] not in operators and tokens[i+1] != ')'
                    next_is_not = tokens[i+1] == 'NOT'
                    next_is_open_paren = tokens[i+1] == '('
                    current_is_close_paren = token == ')'
                    
                    # 情况1: 两个操作数相邻 "a b" -> "a AND b"
                    # 情况2: 操作数后接 NOT "a NOT b" -> 已经在逻辑中，不需要额外加 AND，因为 NOT 是前缀操作符，但这里作为二元连接时通常写作 AND NOT，或者一元。
                    # 为了简化，我们假设用户输入的 "A NOT B" 是合法的。
                    # 但如果用户输入 "A (B)" -> "A AND (B)"
                    
                    should_insert_and = (
                        (current_is_operand and next_is_operand) or
                        (current_is_operand and next_is_open_paren) or
                        (current_is_close_paren and next_is_operand) or
                        (current_is_close_paren and next_is_open_paren) or
                        (current_is_operand and next_is_not) # 支持 "a NOT b" -> "a AND NOT b" 逻辑
                    )
                    
                    if should_insert_and:
                        final_tokens.append('AND')
                        
            return final_tokens
            
        except ValueError:
            # shlex 解析错误（如引号不匹配），直接按空格拆分
            return query.split()

    def _shunting_yard(self, tokens):
        """
        将中缀 token 列表转换为后缀列表 (RPN, Reverse Polish Notation)。
        使用 Shunting Yard 算法。
        
        Args:
            tokens (list): 中缀 Token 列表。
            
        Returns:
            list: 后缀 Token 列表。
        """
        output_queue = []
        operator_stack = []
        
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        
        for token in tokens:
            if token in precedence:
                while (operator_stack and operator_stack[-1] != '(' and 
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                # 操作数 (关键词)
                output_queue.append(token)
        
        while operator_stack:
            output_queue.append(operator_stack.pop())
            
        return output_queue

    def _evaluate_postfix(self, postfix):
        """
        评估后缀表达式，返回 boolean mask。
        
        Args:
            postfix (list): 后缀 Token 列表。
            
        Returns:
            pd.Series: 布尔掩码，长度与 self.df 相同。
        """
        stack = []
        
        for token in postfix:
            if token == 'AND':
                if len(stack) < 2: continue
                val2 = stack.pop()
                val1 = stack.pop()
                stack.append(val1 & val2)
            elif token == 'OR':
                if len(stack) < 2: continue
                val2 = stack.pop()
                val1 = stack.pop()
                stack.append(val1 | val2)
            elif token == 'NOT':
                if len(stack) < 1: continue
                val = stack.pop()
                stack.append(~val)
            else:
                # 这是一个关键词，计算其 mask
                stack.append(self._get_term_mask(token))
                
        return stack[0] if stack else pd.Series([False] * len(self.df), index=self.df.index)

    def _get_term_mask(self, term):
        """
        为单个关键词生成 mask。
        支持通配符 *
        支持短语 (已经在 tokenize 阶段处理为保留空格的字符串)
        
        Args:
            term (str): 搜索词或短语。
            
        Returns:
            pd.Series: 布尔掩码。
        """
        # 转义正则特殊字符，除了 *
        if '*' in term:
            # 处理通配符: embedd* -> \bembedd\w*\b
            # 先转义其他字符
            safe_term = re.escape(term)
            # 将转义后的 \* 替换回 .*? (非贪婪匹配) 或者 \w* (单词字符)
            # 这里我们假设 * 代表任意单词字符序列
            pattern_str = safe_term.replace(r'\*', r'\w*')
            pattern = rf'\b{pattern_str}\b'
        else:
            # 普通词或短语: 增加单词边界
            pattern = rf'\b{re.escape(term)}\b'
            
        # 在指定范围内搜索
        if self.scope == "仅标题":
            mask = self.df['title'].str.contains(pattern, case=False, na=False)
        else:
            mask = (self.df['title'].str.contains(pattern, case=False, na=False) | 
                    self.df['abstract'].str.contains(pattern, case=False, na=False))
            
        return mask

    def _fallback_search(self, query):
        """
        如果解析失败，使用简单的 AND 搜索。
        
        Args:
            query (str): 原始查询字符串。
            
        Returns:
            pd.Series: 布尔掩码。
        """
        terms = query.split()
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        for term in terms:
            term_mask = self._get_term_mask(term)
            mask &= term_mask
        return mask
