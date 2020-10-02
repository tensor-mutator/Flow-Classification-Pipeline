__all__ = ["InvalidLossError", "InvalidOptimizerError", "InvalidMetricError"]

class InvalidLossError(Exception):

      def __init__(self, msg: str) -> None:
          super(InvalidLossError, self).__init__(msg)

class InvalidOptimizerError(Exception):

      def __init__(self, msg: str) -> None:
          super(InvalidOptimizerError, self).__init__(msg)

class InvalidMetricError(Exception):

      def __init__(self, msg: str) -> None:
          super(InvalidMetricError, self).__init__(msg)
