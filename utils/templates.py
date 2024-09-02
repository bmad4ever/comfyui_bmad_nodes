class ComboWrapperNode(type):
    """
    requires attributes: TYPE_NAME ; OPTIONS_LIST
    CATEGORY is not set by the wrapper, must be set in the class.
    """
    def __new__(cls, name, bases, attrs):
        type_name = attrs["TYPE_NAME"].upper()
        arg_name = attrs["TYPE_NAME"].lower()
        options_list = attrs["OPTIONS_LIST"]
        default_option = options_list[0]

        attrs['RETURN_TYPES'] = (type_name,)

        def INPUT_TYPES(cls):
            return {"required": {arg_name: (options_list, {"default": default_option})}}

        def ret_combo(self, **kwargs):
            return (kwargs[attrs["TYPE_NAME"].lower()], )

        attrs["FUNCTION"] = "ret_combo"
        attrs["ret_combo"] = ret_combo
        attrs["INPUT_TYPES"] = classmethod(INPUT_TYPES)
        return super().__new__(cls, name, bases, attrs)
