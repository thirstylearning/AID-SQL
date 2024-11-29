from typing import List
from collections import Set

import sqlparse
import Levenshtein

from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from sql_metadata import Parser

from utils.utils import get_schema_flat_list_from_db_schema


def __DUMP_extract_identifiers(token_list):
    """Extract table and column names from the token list."""
    assert False # This function is not used
    # identifiers = []
    # for token in token_list:
    #     if isinstance(token, IdentifierList):
    #         for identifier in token.get_identifiers():
    #             identifiers.append(identifier.get_real_name())
    #     elif isinstance(token, Identifier):
    #         identifiers.append(token.get_real_name())
    #     elif token.ttype is Keyword:
    #         continue
    #     elif token.is_group:
    #         identifiers.extend(extract_identifiers(token))
    # return identifiers

def extract_tables_and_columns(sql: str):
    tables = set()
    columns = set()
    parser = Parser(sql)
    parsed_columns = parser.columns
    parsed_tables = parser.tables
    
    for c in parsed_columns:
        columns.add(c)
    
    for t in parsed_tables:
        tables.add(t)
        
    return list(tables) + list(columns)
    
    


def modify_query(sql: str, schemas: List[str]):
    """Parse the SQL query, modify table and column names if not in schemas, and reconstruct the query."""

    # Extract table and column names
    identifiers_list = extract_tables_and_columns(sql)

    # Create a mapping of original to modified names
    modified_names = {}
    for name in identifiers_list:
        if name in schemas:
            continue

        for schema_name in schemas:
            edit_distance: int = Levenshtein.distance(name, schema_name)
            if edit_distance <= 3:
                modified_names[name] = schema_name

    print("\n\n\n")
    print(modified_names)
    # Reconstruct the SQL query with modified names
    modified_query = sql
    print(modified_query)
    for original, modified in modified_names.items():
        modified_query = modified_query.replace(original, modified)
        print(modified_query)

    return modified_query




def main():
    # Example usage
    sql_query = "SELECT Money_Rank FROM Poker_Player ORDER BY Earnings DESC LIMIT 1"
    db_schema = [
        {
            "table_name_original": "poker_player",
            "table_name": "poker player",
            "column_names": [
                "poker player id",
                "people id",
                "final table made",
                "best finish",
                "money rank",
                "earnings",
            ],
            "column_names_original": [
                "poker_player_id",
                "people_id",
                "final_table_made",
                "best_finish",
                "money_rank",
                "earnings",
            ],
            "column_types": ["number", "number", "number", "number", "number", "number"],
            "db_contents": [[], [], [], [], [], []],
        },
        {
            "table_name_original": "people",
            "table_name": "people",
            "column_names": ["people id", "nationality", "name", "birth date", "height"],
            "column_names_original": ["people_id", "nationality", "name", "birth_date", "height"],
            "column_types": ["number", "text", "text", "text", "number"],
            "db_contents": [[], [], [], [], []],
        },
    ]

    modified_query = modify_query(sql_query, get_schema_flat_list_from_db_schema(db_schema))


if __name__ == "__main__":
    main()