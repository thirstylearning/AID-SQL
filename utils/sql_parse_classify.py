import sqlparse

from typing import List, Tuple


def query_has_join(sql_query: str) -> bool:
    """
    assume sql_query argument is lowercase.
    """
    # assert sql_query is lowercase
    assert sql_query == sql_query.lower()
    parsed_sql = sqlparse.parse(sql_query)
    for statement in parsed_sql:
        for token in statement.tokens:
            # # recursively check if JOIN in grouped tokens
            # if token.is_group:
            #     if query_has_join(token.value):
            #         return True

            # check if the token value is JOIN
            # if token.value.contains(' join'):
            if 'join' in token.value:
                return True
    return False


def contain_subquery(tokens: sqlparse.sql.TokenList) -> bool:
    for token in tokens:
        if isinstance(token, sqlparse.sql.Parenthesis) and is_subquery(token):
            return True
        elif token.is_group:
            has_subquery: bool = contain_subquery(token.tokens)
            if has_subquery:
                return True


def contain_set_operation(parsed_sql: sqlparse.sql.TokenList) -> bool:
    SET_OPERATIONS_ENUM = ['union', 'intersect', 'except']
    for token in parsed_sql.tokens:
        if token.ttype in sqlparse.tokens.Keyword and any(op in token.value for op in SET_OPERATIONS_ENUM):
            return True
    return False


def query_has_subquery_or_set_op(sql_query: str) -> Tuple[bool, List[str]]:
    """
    assume sql_query argument is lowercase.
    if query has subquery, return True with extracted subquery str list
    :return: Tuple[bool, List[str]], bool: has_subquery, List[str]: subquery list
    """
    assert sql_query == sql_query.lower()
    parsed_sql = sqlparse.parse(sql_query)[0]  # only one statement
    set_op_flag: bool = False
    contain_subquery_flag: bool = False
    if contain_set_operation(parsed_sql):
        set_op_flag = True
    if contain_subquery(parsed_sql):
        contain_subquery_flag = True
    subquery_list = []
    if contain_subquery_flag:
        subquery_list = extract_subquery(sql_query)
    return set_op_flag or contain_subquery_flag, subquery_list


def strip_parenthesis_wrapped_sql(sql_query: str) -> str:
    """
    assume input sql_query is wrapped by parenthesis.
    """
    sql_query = sql_query.strip()
    assert sql_query[0] == '(' and sql_query[-1] == ')'
    sql_query = sql_query[1:-1].strip()
    return sql_query


def is_subquery(parenthesis):
    """check if the parenthesis token is a sub-query"""
    return any(token.match(sqlparse.tokens.DML, 'select') for token in parenthesis.tokens)


def extract_subquery(sql_query: str) -> List[str]:
    """
    assume sql_query argument is lowercase, and has subquery.
    """
    subqueries = []
    parsed_sql = sqlparse.parse(sql_query)[0]  # only one statement

    def traverse(tokens: sqlparse.sql.TokenList):
        for token in tokens:
            if isinstance(token, sqlparse.sql.Parenthesis) and is_subquery(token):
                subqueries.append(strip_parenthesis_wrapped_sql(token.value))
            elif token.is_group:
                traverse(token.tokens)

    traverse(parsed_sql)

    return subqueries


if __name__ == '__main__':
    # for test
    sql_query1 = "SELECT * FROM (SELECT * FROM table1) t1 UNION SELECT * FROM table2"
    sql_query1 = sql_query1.lower()
    print(query_has_subquery_or_set_op(sql_query1))

    sql_query2 = "SELECT * FROM table1 INNER JOIN table2 ON table1.id = table2.id"
    sql_query2 = sql_query2.lower()
    print(query_has_join(sql_query2))

    sql_statement = """
    SELECT * FROM users WHERE id IN (
        SELECT user_id FROM orders WHERE order_date > '2022-01-01'
    ) AND EXISTS (
        SELECT 1 FROM products WHERE products.id = users.favorite_product_id
    )
    """
    subqueries = extract_subquery(sql_statement.lower())
    print(subqueries)

    bool_v, subqueries = query_has_subquery_or_set_op(sql_statement.lower())
    print(bool_v)
    print(subqueries)
