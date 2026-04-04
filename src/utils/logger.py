import logging


class ColorFormatter(logging.Formatter):
	RESET = "\033[0m"
	LEVEL_COLORS = {
		logging.DEBUG: "\033[36m",     # Cyan
		logging.INFO: "\033[32m",      # Green
		logging.WARNING: "\033[33m",   # Yellow
		logging.ERROR: "\033[31m",     # Red
		logging.CRITICAL: "\033[35m",  # Magenta
	}

	def __init__(
		self,
		fmt: str,
		datefmt: str | None = None,
		use_color: bool = True,
		level_right_pad: int = 10,
	):
		super().__init__(fmt=fmt, datefmt=datefmt)
		self.use_color = use_color
		self.level_right_pad = level_right_pad

	def format(self, record: logging.LogRecord) -> str:
		record.level_suffix = " " * max(1, self.level_right_pad - len(record.levelname))
		message = super().format(record)
		if not self.use_color:
			return message

		color = self.LEVEL_COLORS.get(record.levelno)
		if not color:
			return message

		return f"{color}{message}{self.RESET}"

class Logger:
	def __init__(
		self,
		name: str = "my_logger",
		log_level: int = logging.DEBUG,
		format: str = "%(asctime)s [%(levelname)s]%(level_suffix)s: %(message)s",
		date_format: str = "%H:%M:%S",
		use_color: bool = True,
		level_right_pad: int = 10,
	):
		self.logger = logging.getLogger(name)
		self.logger.setLevel(log_level)
		self.logger.propagate = False

		self.formatter = ColorFormatter(
			format,
			datefmt=date_format,
			use_color=use_color,
			level_right_pad=level_right_pad,
		)

		if not self.logger.handlers:
			self.handler = logging.StreamHandler()
			self.handler.setFormatter(self.formatter)
			self.logger.addHandler(self.handler)
		else:
			for handler in self.logger.handlers:
				handler.setFormatter(self.formatter)

	def debug(self, msg: str, *args, **kwargs):
		self.logger.debug(msg, *args, **kwargs)

	def info(self, msg: str, *args, **kwargs):
		self.logger.info(msg, *args, **kwargs)

	def warn(self, msg: str, *args, **kwargs):
		self.logger.warning(msg, *args, **kwargs)

	def error(self, msg: str, *args, **kwargs):
		self.logger.error(msg, *args, **kwargs)
	
	def critical(self, msg: str, *args, **kwargs):
		self.logger.critical(msg, *args, **kwargs)
	
