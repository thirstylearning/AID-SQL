from typing import Dict, List


def get_tc_dict_from_db_schema(db_schema: List[Dict]) -> Dict:
    """
    get key-table, value-columns dict, from db_schema field
    """
    table_cols_dict: Dict = {}
    for table in db_schema:
        table_cols_dict[table['table_name_original']] = [col_name for col_name in table['column_names_original']]

    return table_cols_dict


def get_tc_schema_link_dict_from_db_schema(db_schema: List[Dict],
                                           table_labels: List[int],
                                           column_labels: List[List[int]]) -> Dict:
    """
    get key-table, value-columns dict, from db_schema field with used tables and columns
    """
    tc_schema_link_dict = {}
    for table_idx, (table, table_label) in enumerate(zip(db_schema, table_labels)):
        if table_label == 0:
            continue

        used_col_names: List[str] = []
        for col_name, col_label in zip(table['column_names_original'], column_labels[table_idx]):
            if col_label == 0:
                continue

            used_col_names.append(col_name)

        tc_schema_link_dict[table['table_name_original']] = used_col_names

    return tc_schema_link_dict


def get_tc_sequence_from_tc_dict(tc_dict: Dict[str, List[str]]) -> str:
    """
    get BASIC_REPR liked db_schema sequence from tc_dict
    """
    table_sequence_list: List[str] = []
    for table_name, cols_list in tc_dict.items():
        columns_sequence: str = ",".join(cols_list)
        table_sequence: str = f"Table {table_name}, columns = [{columns_sequence}]"
        table_sequence_list.append(table_sequence)

    return "\n".join(table_sequence_list)


def get_tc_schema_link_dot_sequence_from_schema_link_dict(tc_schema_link_dict: Dict[str, List[str]]) -> str:
    """
    dot seq like: table1.col1, table2.col2
    """
    t_dot_c_list: List[str] = []
    for table_name, cols_list in tc_schema_link_dict.items():
        for col_name in cols_list:
            t_dot_c_list.append(f"{table_name}.{col_name}")

    return ",".join(t_dot_c_list)
