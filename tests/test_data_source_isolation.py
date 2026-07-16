"""数据源隔离纪律的单测：除 data.py 外，任何模块都不允许直接 import akshare
或 WindPy —— 上层只能通过 DataAdapter 接口 + 标准面板 schema 拿数据，这样以后
从 akshare 换到 Wind（或任何新数据源）只需要改 get_adapter() 的一个参数。
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASIS_MODEL_DIR = PROJECT_ROOT / "basis_model"

#: 允许直接引用具体数据源 SDK 的唯一文件
ALLOWED_FILE = "data.py"

#: 禁止在 data.py 以外出现的数据源 SDK 模块名
FORBIDDEN_MODULES = {"akshare", "WindPy"}


def _imported_module_names(py_file: Path) -> set[str]:
    """解析一个 .py 文件的 AST，收集所有 import / from-import 的顶层模块名。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_only_data_py_imports_concrete_data_sources():
    violations: dict[str, set[str]] = {}
    for py_file in BASIS_MODEL_DIR.glob("*.py"):
        if py_file.name == ALLOWED_FILE:
            continue
        imported = _imported_module_names(py_file)
        hit = imported & FORBIDDEN_MODULES
        if hit:
            violations[py_file.name] = hit

    assert not violations, (
        f"以下模块直接 import 了具体数据源 SDK，违反纪律（只允许 data.py 触碰）: "
        f"{violations}"
    )


def test_data_py_is_the_one_place_allowed_to_import_akshare():
    data_py = BASIS_MODEL_DIR / "data.py"
    imported = _imported_module_names(data_py)
    assert "akshare" in imported, "data.py 应当是引用 akshare 的地方（AkShareAdapter）"
