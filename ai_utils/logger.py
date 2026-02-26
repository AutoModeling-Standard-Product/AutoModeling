"""
统一日志系统模块 - 风险建模项目日志管理

该模块提供统一的日志配置和管理功能，支持：
- 控制台和文件双输出
- 日志级别动态调整
- 结构化日志格式
- 时间戳和调用者信息

使用指南:
    >>> from ai_utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("模型训练开始")
    >>> logger.warning("检测到缺失值")
    >>> logger.error("训练失败", exc_info=True)

Version: 1.0.0
Author: AutoModeling Team
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# 全局日志配置字典
_loggers: dict = {}


def setup_logging(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> None:
    """
    设置全局日志系统
    
    创建日志目录，配置日志格式和处理器。该函数应在程序启动时调用一次。
    
    Args:
        log_dir (str): 日志文件存储目录，默认为"./logs"
        log_level (int): 日志级别，默认为 logging.INFO
            - logging.DEBUG: 详细调试信息
            - logging.INFO: 常规信息（推荐）
            - logging.WARNING: 警告信息
            - logging.ERROR: 错误信息
        log_to_file (bool): 是否将日志输出到文件，默认为 True
        log_to_console (bool): 是否将日志输出到控制台，默认为 True
    
    Returns:
        None
    
    Raises:
        OSError: 当日志目录创建失败时
        ValueError: 当日志级别无效时
    
    Business Context:
        统一日志系统是生产环境监控的基础：
        - 便于问题排查和审计追踪
        - 支持日志聚合和分析（如 ELK Stack）
        - 符合监管要求的日志保存规范
        - 日志文件应定期轮转和归档
        
        日志级别选择建议：
        - 开发环境: DEBUG
        - 测试环境: INFO
        - 生产环境: INFO 或 WARNING
        
        日志内容要求：
        - 包含业务上下文（机构、时间、操作）
        - 敏感信息脱敏（客户ID、身份证号）
        - 关键决策点记录（参数选择、阈值设定）
    
    Example:
        >>> # 设置日志系统
        >>> setup_logging(
        ...     log_dir="./logs",
        ...     log_level=logging.INFO,
        ...     log_to_file=True,
        ...     log_to_console=True
        ... )
        >>> 
        >>> # 获取并使用日志器
        >>> logger = get_logger(__name__)
        >>> logger.info("系统启动完成")
    
    Note:
        - 日志文件命名格式：YYYYMMDD.log
        - 每天创建新日志文件
        - 已存在的日志文件会追加写入
        - 建议定期清理旧日志文件（保留30-90天）
    """
    # 验证日志级别
    if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError(
            f"无效的日志级别: {log_level}. "
            "必须为 logging.DEBUG/INFO/WARNING/ERROR/CRITICAL"
        )
    
    # 创建日志目录
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"无法创建日志目录 {log_dir}: {str(e)}") from e
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 定义日志格式
    log_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_to_file:
        log_file = log_path / f"{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志器
    
    返回指定名称的日志器实例。如果日志器已存在则返回缓存的实例，
    否则创建新实例。日志器继承全局配置。
    
    Args:
        name (str): 日志器名称，通常使用 __name__ 获取模块名
            - 模块级: "ai_utils.optuna_lgb"
            - 类级: "ai_utils.optuna_lgb.OptunaLGB"
            - 推荐使用 __name__ 以自动获取正确名称
    
    Returns:
        logging.Logger: 配置好的日志器实例
            - 支持 debug/info/warning/error/critical 方法
            - 支持格式化字符串: logger.info("训练轮次: %d", epoch)
            - 支持异常堆栈: logger.error("失败", exc_info=True)
    
    Raises:
        RuntimeError: 当未调用 setup_logging 时
    
    Business Context:
        日志器命名规范：
        - 使用模块全路径作为名称
        - 便于日志过滤和分组分析
        - 符合 Python logging 最佳实践
        
        典型使用场景：
        - 模块入口: logger.info("模块加载")
        - 数据处理: logger.debug("处理数据: shape=%s", data.shape)
        - 参数验证: logger.warning("参数超出范围: value=%d", value)
        - 错误处理: logger.error("操作失败", exc_info=True)
    
    Example:
        >>> # 模块顶部初始化
        >>> from ai_utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> 
        >>> # 在函数中使用
        >>> def train_model(X, y):
        ...     logger.info("开始训练模型，样本数: %d", len(X))
        ...     try:
        ...         # 训练代码
        ...         logger.info("训练完成，AUC: %.4f", auc)
        ...     except Exception as e:
        ...         logger.error("训练失败: %s", str(e), exc_info=True)
        ...         raise
    
    Note:
        - 日志器是单例模式，同一名称返回同一实例
        - 继承根日志器的配置（级别、处理器）
        - 支持多进程安全（但需注意日志文件锁）
        - 建议在模块导入时调用 get_logger，而非在函数内重复调用
    """
    # 检查是否已初始化全局日志系统
    if name in _loggers:
        return _loggers[name]
    
    # 获取根日志器
    root_logger = logging.getLogger()
    
    # 如果根日志器没有处理器，提示用户调用 setup_logging
    if not root_logger.handlers:
        raise RuntimeError(
            "日志系统未初始化。请先调用 setup_logging() 函数。"
            "例如: setup_logging(log_dir='./logs', log_level=logging.INFO)"
        )
    
    # 创建命名日志器
    logger = logging.getLogger(name)
    _loggers[name] = logger
    
    return logger


# ==============================================================================
# 快捷函数 - 简化日志记录
# ==============================================================================

def log_info(message: str, name: Optional[str] = None) -> None:
    """
    快捷记录 INFO 级别日志
    
    Args:
        message (str): 日志消息
        name (Optional[str]): 日志器名称，默认使用调用者的 __name__
    """
    logger = get_logger(name or __name__)
    logger.info(message)


def log_warning(message: str, name: Optional[str] = None) -> None:
    """
    快捷记录 WARNING 级别日志
    
    Args:
        message (str): 日志消息
        name (Optional[str]): 日志器名称，默认使用调用者的 __name__
    """
    logger = get_logger(name or __name__)
    logger.warning(message)


def log_error(message: str, exc_info: bool = False, name: Optional[str] = None) -> None:
    """
    快捷记录 ERROR 级别日志
    
    Args:
        message (str): 日志消息
        exc_info (bool): 是否记录异常堆栈，默认为 False
        name (Optional[str]): 日志器名称，默认使用调用者的 __name__
    """
    logger = get_logger(name or __name__)
    logger.error(message, exc_info=exc_info)


def log_debug(message: str, name: Optional[str] = None) -> None:
    """
    快捷记录 DEBUG 级别日志
    
    Args:
        message (str): 日志消息
        name (Optional[str]): 日志器名称，默认使用调用者的 __name__
    """
    logger = get_logger(name or __name__)
    logger.debug(message)


__all__ = [
    'setup_logging',
    'get_logger',
    'log_info',
    'log_warning',
    'log_error',
    'log_debug'
]