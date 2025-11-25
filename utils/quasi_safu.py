import math
import threading
import multiprocessing
from simpleeval import SimpleEval, EvalWithCompoundTypes, NameNotDefined
from collections.abc import Callable
from typing import Set
import signal
import inspect
import re


# Comprehensive list of dangerous keywords that could lead to security issues
# IMPORTANT NOTE:
#    some of methods won't execute with simpleeval, nevertheless, redundancy won't only help with the environment safety
#    it will also allow outputting a more user-friendly message indicating the problem,
#    so that a workflow containing such keywords can be reported.
DANGEROUS_KEYWORDS = {
    # File I/O operations
    'open', 'read', 'write', 'close', 'file', 'load', 'save', 'dump',
    'imread', 'imwrite', 'imload', 'imsave',
    'readlines', 'readline', 'writelines', 'writeline',
    'append', 'truncate', 'seek', 'tell',

    # File system operations
    'remove', 'unlink', 'rmdir', 'mkdir', 'makedirs', 'removedirs',
    'rename', 'replace', 'move', 'copy', 'copytree', 'rmtree',
    'chdir', 'chmod', 'chown', 'link', 'symlink',
    'listdir', 'scandir', 'walk', 'glob',

    # System/OS operations
    'system', 'popen', 'exec', 'eval', 'compile', 'execfile',
    'call', 'check_call', 'check_output', 'run', 'spawn',
    'fork', 'kill', 'terminate', 'communicate',
    'shell', 'command', 'cmd', 'powershell', 'bash',

    # Import and module manipulation
    'import', '__import__', 'importlib', 'reload',
    'module', 'package',

    # Dunder/magic methods (potential exploits)
    '__class__', '__bases__', '__subclasses__', '__mro__',
    '__globals__', '__builtins__', '__dict__', '__code__',
    '__init__', '__new__', '__call__', '__getattribute__',
    '__setattr__', '__delattr__', '__getitem__', '__setitem__',
    '__import__', '__loader__', '__package__', '__spec__',

    # Network operations
    'socket', 'connect', 'bind', 'listen', 'accept', 'send', 'recv',
    'request', 'urlopen', 'urlretrieve', 'download', 'upload',
    'http', 'https', 'ftp', 'ssh', 'telnet',
    'get', 'post', 'put', 'delete', 'patch',

    # Database operations
    'execute', 'executemany', 'cursor', 'commit', 'rollback',
    'query', 'select', 'insert', 'update', 'delete', 'drop',
    'create', 'alter', 'grant', 'revoke',

    # Serialization/deserialization (can execute code)
    'pickle', 'unpickle', 'marshal', 'shelve',
    'yaml', 'json', 'toml', 'xml',

    # Video I/O (OpenCV)
    'VideoWriter', 'VideoCapture', 'VideoReader',
    'FileStorage', 'FileNode',

    # Memory/process manipulation
    'mmap', 'memoryview', 'buffer', 'bytearray',
    'ctypes', 'cffi', 'array',

    # Environment/configuration access
    'environ', 'getenv', 'setenv', 'putenv', 'unsetenv',
    'config', 'settings', 'preferences',

    # Reflection/introspection
    'getattr', 'setattr', 'delattr', 'hasattr',
    'vars', 'dir', 'globals', 'locals',
    'type', 'isinstance', 'issubclass',

    # Code generation/execution
    'input', 'raw_input', 'compile', 'ast',
    'codecs', 'decode', 'encode',

    # Process/thread creation
    'Thread', 'Process', 'Pool', 'Queue',
    'Popen', 'subprocess',
}


def sanitize_expression(expression: str,
                        custom_dangerous_keywords: Set[str] | None = None,
                        case_sensitive: bool = False,
                        max_iterations: int = 100) -> tuple[str, bool, list[str]]:
    """
    Sanitize an expression by removing dangerous keywords with onion attack protection.

    This function repeatedly removes dangerous keywords until none are found, protecting
    against obfuscation techniques like "imimreadread" which becomes "imread" after one pass.

    Args:
        expression: The expression string to sanitize
        custom_dangerous_keywords: Additional dangerous keywords to check for (optional)
        case_sensitive: If True, matching is case-sensitive (default: False)
        max_iterations: Maximum number of cleaning iterations to prevent infinite loops (default: 100)

    Returns:
        A tuple of (sanitized_expression, was_modified, list_of_found_keywords)
        - sanitized_expression: The cleaned expression
        - was_modified: True if any dangerous keywords were found and removed
        - list_of_found_keywords: List of all dangerous keywords that were found

    Examples:
        >>> sanitize_expression("cv.imread('file.jpg')")
        ("cv.('file.jpg')", True, ['imread'])

        >>> sanitize_expression("imimreadread")  # Onion attack
        ("", True, ['imread'])

        >>> sanitize_expression("safe_expression * 2")
        ("safe_expression * 2", False, [])
    """
    # Combine default and custom keywords
    all_keywords = DANGEROUS_KEYWORDS.copy()
    if custom_dangerous_keywords:
        all_keywords.update(custom_dangerous_keywords)

    # Sort keywords by length (longest first) to catch longer patterns first
    # This helps with patterns like "__import__" before matching "import"
    sorted_keywords = sorted(all_keywords, key=len, reverse=True)

    original_expression = expression
    found_keywords = []
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1
        modified_this_iteration = False

        for keyword in sorted_keywords:
            if case_sensitive:
                pattern = re.escape(keyword)
            else:
                # Create case-insensitive pattern
                pattern = ''.join(f'[{c.upper()}{c.lower()}]' if c.isalpha() else re.escape(c)
                                  for c in keyword)

            # Check if keyword exists in the expression
            if re.search(pattern, expression):
                # Remove all occurrences of this keyword
                new_expression = re.sub(pattern, '', expression)

                if new_expression != expression:
                    modified_this_iteration = True
                    if keyword not in found_keywords:
                        found_keywords.append(keyword)
                    expression = new_expression

        # If no modifications were made this iteration, we're done
        if not modified_this_iteration:
            break

    # Check if we hit max iterations (potential infinite loop or very nested attack)
    if iteration_count >= max_iterations:
        raise ValueError(
            f"Maximum iterations ({max_iterations}) reached during sanitization. "
            f"This might indicate a very complex obfuscation attempt."
        )

    was_modified = (expression != original_expression)

    return expression, was_modified, found_keywords


def validate_expression_safety(expression: str,
                               custom_dangerous_keywords: Set[str] | None = None,
                               case_sensitive: bool = False,
                               raise_on_dangerous: bool = True) -> bool:
    """
    Validate if an expression is safe (contains no dangerous keywords).

    This is a convenience wrapper around sanitize_expression that focuses on validation.

    Args:
        expression: The expression to validate
        custom_dangerous_keywords: Additional dangerous keywords to check for (optional)
        case_sensitive: If True, matching is case-sensitive (default: False)
        raise_on_dangerous: If True, raises ValueError when dangerous keywords found (default: True)

    Returns:
        True if expression is safe, False otherwise

    Raises:
        ValueError: If raise_on_dangerous=True and dangerous keywords are found

    Examples:
        >>> validate_expression_safety("x + y * 2")
        True

        >>> validate_expression_safety("open('file.txt')")
        ValueError: Dangerous keywords found: ['open', 'file']

        >>> validate_expression_safety("open('file.txt')", raise_on_dangerous=False)
        False
    """
    _, was_modified, found_keywords = sanitize_expression(
        expression,
        custom_dangerous_keywords=custom_dangerous_keywords,
        case_sensitive=case_sensitive
    )

    if was_modified and raise_on_dangerous:
        raise ValueError(
            f"Dangerous keywords found in expression: {found_keywords}\n"
            f"Original expression: {expression}"
        )

    return not was_modified


def filter_expression_names(validator: Callable[[str], bool], expression: str, src_object: any) -> Set[str]:
    """Extract valid names from expression that pass the validator function."""
    import re
    # Extract all potential identifiers from the expression
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    obj_props = dir(src_object)
    return {name for name in identifiers if validator(name) and name in obj_props}


class TimeoutException(Exception):
    """Custom exception for timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout (Unix only)."""
    raise TimeoutException("Expression evaluation timed out")


# ============================================================================
# MULTIPROCESSING APPROACH (Stronger DoS protection, no instance methods)
# ============================================================================

def _evaluate_in_process(expr: str, names_dict: dict, funcs_dict: dict,
                         math_funcs: dict, use_compound_types: bool, result_queue):
    """
    Helper function to run in separate process.
    Must be at module level for pickling to work.

    Note: Only works with picklable objects (no instance methods).
    We can't pass custom evaluator instances, so we use a flag instead.
    """
    try:
        # Create the evaluator based on the flag
        if use_compound_types:
            evaluator = EvalWithCompoundTypes()
        else:
            evaluator = SimpleEval()

        evaluator.names = {**names_dict, **math_funcs}
        evaluator.functions = funcs_dict

        result = evaluator.eval(expr)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', str(e), type(e).__name__))


def _evaluate_with_multiprocessing(expression: str, names: dict[str, any],
                                   functions: dict[str, Callable],
                                   math_funcs: dict[str, any],
                                   use_compound_types: bool,
                                   timeout: float) -> any:
    """
    Evaluate with multiprocessing for maximum DoS protection.
    Process can be forcefully killed if it exceeds timeout.
    """
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_evaluate_in_process,
        args=(expression, names, functions, math_funcs, use_compound_types, result_queue)
    )
    process.start()

    process.join(timeout=timeout)

    if process.is_alive():
        # Process is still running - terminate it
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(
            f"Expression evaluation exceeded {timeout} seconds "
            f"(killed via multiprocessing)"
        )

    if result_queue.empty():
        raise RuntimeError("Evaluation process terminated without returning a result")

    status, *data = result_queue.get()

    if status == 'success':
        return data[0]
    else:
        error_msg, error_type = data
        if error_type == 'NameNotDefined':
            raise NameNotDefined(error_msg)
        else:
            raise ValueError(f"Evaluation error: {error_msg}")


# ============================================================================
# THREADING APPROACH (Weaker DoS protection, supports instance methods)
# ============================================================================

def _evaluate_with_threading(expression: str, names: dict[str, any],
                             functions: dict[str, Callable],
                             math_funcs: dict[str, any],
                             evaluator_instance: SimpleEval | None,
                             use_compound_types: bool,
                             timeout: float) -> any:
    """
    Evaluate with threading. Supports instance methods and custom evaluators
    but weaker timeout protection.
    """
    result_container = {'result': None, 'error': None, 'completed': False}

    def _evaluate():
        """Helper function to run in thread."""
        try:
            # Use custom evaluator if provided, otherwise create one
            if evaluator_instance is not None:
                evaluator = evaluator_instance
            elif use_compound_types:
                evaluator = EvalWithCompoundTypes()
            else:
                evaluator = SimpleEval()

            evaluator.names = {**names, **math_funcs}
            evaluator.functions = functions

            result = evaluator.eval(expression)
            result_container['result'] = result
            result_container['completed'] = True
        except Exception as e:
            result_container['error'] = e
            result_container['completed'] = True

    # For Unix systems, try to use signal-based timeout for better CPU protection
    use_signal = (
            hasattr(signal, 'SIGALRM') and
            threading.current_thread() is threading.main_thread()
    )

    if use_signal:
        # Set up signal-based timeout (Unix only, main thread only)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)

        try:
            _evaluate()
            signal.setitimer(signal.ITIMER_REAL, 0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        except TimeoutException:
            signal.signal(signal.SIGALRM, old_handler)
            raise TimeoutError(
                f"Expression evaluation exceeded {timeout} seconds "
                f"(signal-based timeout)"
            )
    else:
        # Fall back to threading-based timeout
        thread = threading.Thread(target=_evaluate, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Expression evaluation exceeded {timeout} seconds "
                f"(thread-based timeout - thread continues as daemon)"
            )

    if not result_container['completed']:
        raise TimeoutError(f"Expression evaluation exceeded {timeout} seconds")

    if result_container['error'] is not None:
        error = result_container['error']
        if isinstance(error, NameNotDefined):
            raise error
        else:
            raise ValueError(f"Evaluation error: {str(error)}")

    return result_container['result']


# ============================================================================
# SMART DISPATCHER (Chooses best evaluation method)
# ============================================================================

def _is_picklable(obj) -> bool:
    """
    Check if an object is truly picklable for multiprocessing.

    This is more thorough than just trying pickle.dumps() because on Windows
    with spawn, objects can pickle in the parent but fail in the child.
    """
    import pickle

    # First, try basic pickling
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError):
        return False

    # Additional checks for common unpicklable cases
    if callable(obj):
        # Check if it's a bound method (instance method)
        if inspect.ismethod(obj):
            return False

        # Check if it's defined in __main__ (won't work with spawn on Windows)
        if hasattr(obj, '__module__') and obj.__module__ == '__main__':
            return False

        # Check if it's a lambda
        if obj.__name__ == '<lambda>':
            return False

        # Check for decorators that might cause issues
        if hasattr(obj, '__wrapped__'):
            # Decorated functions might not pickle well
            return _is_picklable(obj.__wrapped__)

    return True


def _can_use_multiprocessing(names: dict[str, any],
                             functions: dict[str, Callable]) -> bool:
    """
    Determine if we can use multiprocessing based on whether all objects are picklable.
    """
    # Check all names
    for name, value in names.items():
        if not _is_picklable(value):
            return False

    # Check all functions
    for name, func in functions.items():
        if not _is_picklable(func):
            return False

    return True


def evaluate_with_timeout(expression: str, names: dict[str, any],
                          functions: dict[str, Callable], math_funcs: dict[str, any],
                          evaluator: SimpleEval | None,
                          use_compound_types: bool,
                          timeout: float, force_threading: bool = False) -> any:
    """
    Intelligently choose between multiprocessing and threading for evaluation.

    Strategy:
    - Multiprocessing (preferred): Stronger DoS protection, can kill runaway processes
      - Cannot use custom evaluator instances (uses use_compound_types flag instead)
    - Threading (fallback): Weaker protection, but supports instance methods, unpicklable
      objects, and custom evaluator instances

    Args:
        expression: The expression string to evaluate
        names: Dictionary of allowed names/variables
        functions: Dictionary of allowed functions
        math_funcs: Dictionary of math functions to make available
        evaluator: Custom SimpleEval instance (only works with threading)
        use_compound_types: If True, use EvalWithCompoundTypes instead of SimpleEval
        timeout: Maximum execution time in seconds
        force_threading: If True, always use threading (for testing)

    Returns:
        The result of the evaluation

    Raises:
        TimeoutError: If evaluation exceeds timeout
        ValueError: If expression is invalid or uses forbidden constructs
    """
    # If a custom evaluator is provided, we must use threading
    must_use_threading = force_threading or evaluator is not None

    # Decide which approach to use
    use_multiprocessing = (
            not must_use_threading and
            _can_use_multiprocessing(names, functions)
    )

    if use_multiprocessing:
        return _evaluate_with_multiprocessing(
            expression, names, functions, math_funcs, use_compound_types, timeout
        )
    else:
        return _evaluate_with_threading(
            expression, names, functions, math_funcs, evaluator, use_compound_types, timeout
        )


# ============================================================================
# PUBLIC API
# ============================================================================

def safe_eval(expression: str,
              source_object: any = None,
              token_validator: Callable[[str], bool] | None = None,
              allowed_functions: dict[str, Callable] | None = None,
              extra_names: dict[str, any] | None = None,
              evaluator: SimpleEval | None = None,
              use_compound_types: bool = False,
              include_math: bool = True,
              timeout: float = 5.0,
              force_threading: bool = False) -> any:
    """
    Safely evaluate an expression with security and DoS protection.

    Automatically chooses the best evaluation method:
    - Multiprocessing: Used when all objects are picklable (stronger DoS protection)
    - Threading: Used for instance methods, unpicklable objects, or custom evaluators (weaker DoS protection)

    Args:
        expression: The expression to evaluate
        source_object: Object to extract allowed attributes/methods from (optional)
        token_validator: Function to validate which tokens from source_object are allowed.
                        Should take a token name and return True if allowed. (optional)
                         The provided token_validator does not need to check that the token belongs to the source_object
                         this is already done by this (safe_eval) function.
                         Also required to avoid pre-computing data that is not needed.
        allowed_functions: Dictionary of allowed functions (e.g., {'min': min, 'max': max})
        extra_names: Additional names/variables to make available in the expression (optional)
        evaluator: Custom SimpleEval or EvalWithCompoundTypes instance (forces threading) (optional)
        use_compound_types: If True, use EvalWithCompoundTypes instead of SimpleEval (default: False)
                           Ignored if evaluator is provided.
        include_math: If True, includes common math functions under 'm' namespace (default: True)
        timeout: Maximum evaluation time in seconds (default: 5.0)
        force_threading: If True, always use threading (for testing/debugging)

    Returns:
        The result of the evaluation

    Raises:
        ValueError: If expression contains invalid tokens
        TimeoutError: If evaluation exceeds timeout

    Note:
        - Custom evaluator instances can only be used with threading (weaker DoS protection)
        - For multiprocessing, use use_compound_types=True instead of passing a custom evaluator

    Examples:
        # Simple evaluation with compound types (uses multiprocessing)
        result = safe_eval("{'a': 1, 'b': 2}", use_compound_types=True)

        # With custom evaluator (forces threading)
        custom_eval = SimpleEval()
        custom_eval.operators[ast.BitXor] = operator.xor
        result = safe_eval("5 ^ 3", evaluator=custom_eval)

        # With instance methods (automatically uses threading)
        result = safe_eval(
            "h_computed()",
            source_object=samples,
            token_validator=lambda t: t.startswith("h_")
        )
    """
    # remove comments and new lines
    expression = re.sub('#.+', '', expression)
    expression = re.sub('\n', '', expression)

    # if potentially dangerous, raise exception and print the dangerous keywords
    validate_expression_safety(expression)

    # Start with empty names dictionary
    names_to_include = {}

    # Make a copy to avoid modifying the original
    allowed_functions = {} if allowed_functions is None else allowed_functions.copy()

    # If source_object is provided, extract allowed attributes
    if source_object is not None:
        if token_validator is None:
            raise ValueError("token_validator must be provided when using source_object")

        # Extract and validate names from expression
        allowed_token_names = filter_expression_names(token_validator, expression, source_object)

        # Separate methods/callables from regular attributes
        for name in allowed_token_names:
            attr = getattr(source_object, name)
            if callable(attr):
                # Add callables to functions dict
                allowed_functions[name] = attr
            else:
                # Add non-callable attributes to names dict
                names_to_include[name] = attr

    # Add any extra names provided
    if extra_names:
        names_to_include.update(extra_names)

    # Build math functions dictionary if requested
    math_functions = {}
    if include_math:
        math_functions = {
            'm': {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'atan2': math.atan2,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'asinh': math.asinh,
                'acosh': math.acosh,
                'atanh': math.atanh,
                'radians': math.radians,
                'degrees': math.degrees,

                'log': math.log,
                'log10': math.log10,
                'log2': math.log2,
                'exp': math.exp,
                'expm1': math.expm1,
                'pow': math.pow,

                'fabs': math.fabs,
                'factorial': math.factorial,
                'gamma': math.gamma,
                'lgamma': math.lgamma,
                'trunc': math.trunc,
                'floor': math.floor,
                'ceil': math.ceil,
                'isfinite': math.isfinite,
                'isinf': math.isinf,
                'isnan': math.isnan,

                'hypot': math.hypot,
                'dist': math.dist,
                'copysign': math.copysign,
                'fmod': math.fmod,
                'remainder': math.remainder,
                'frexp': math.frexp,
                'ldexp': math.ldexp,
                'modf': math.modf,

                # Constants
                'pi': math.pi,
                'e': math.e,
                'tau': math.tau,
                'inf': math.inf,
                'nan': math.nan,
            }
        }

    # Evaluate with intelligent timeout protection
    result = evaluate_with_timeout(
        expression=expression,
        names=names_to_include,
        functions=allowed_functions,
        math_funcs=math_functions,
        evaluator=evaluator,
        use_compound_types=use_compound_types,
        timeout=timeout,
        force_threading=force_threading
    )

    return result