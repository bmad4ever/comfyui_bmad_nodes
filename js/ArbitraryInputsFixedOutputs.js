import { app } from "/scripts/app.js";

var mi_node_types = {
    "VAEEncodeBatch": "IMAGE",
    "CondList (string) Advanced": "CONDITIONING",
    "CLIPEncodeMultipleAdvanced": "STRING",
    "ControlNetHadamard (manual)": "IMAGE",

    "ToMaskList": "MASK",
    "ToImageList": "IMAGE",
    "ToLatentList": "LATENT",
    "ToCondList": "CONDITIONING",
    "ToModelList": "MODEL",
    "ToColorList": "COLOR",
    "ToStringList": "STRING",
    "ToIntList": "INT",
    "ToFloatList": "FLOAT",

    "ExtendMaskList": "MASK",
    "ExtendImageList": "IMAGE",
    "ExtendLatentList": "LATENT",
    "ExtendCondList": "CONDITIONING",
    "ExtendModelList": "MODEL",
    "ExtendColorList": "COLOR",
    "ExtendStringList": "STRING",
    "ExtendIntList": "INT",
    "ExtendFloatList": "FLOAT",
    }


app.registerExtension({
	name: "Comfy.Bmad.ArbitraryInputsFixedOutputs",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if( !(nodeData.name in mi_node_types) ) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
				const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
                this.input_type = mi_node_types[nodeData.name];
                this.var_prefix = this.input_type.toLowerCase() + "_"
				return r;
		};

		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
		    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;

            // add option
            options.unshift(
				{
					content: "update Inputs",
					callback: () => {
    					const new_inputs_len = this.widgets.find(w => w.name === "inputs_len")["value"];
					    const initial_total_inputs_len = this.inputs === undefined? 0 : this.inputs.length;

                        // count number of variable inputs and place fixed inputs at the top
                        let initial_number_of_variable_inputs = 0; // the number of inputs ignoring fixed inputs
                        let index = initial_total_inputs_len - 1;
                        for(let i = initial_total_inputs_len - 1 ; i>= 0; --i)
                        {
                            console.log(`check it: ${this.inputs[index]["name"]} ; ${this.var_prefix}`)
                            if(this.inputs[index]["name"].startsWith(this.var_prefix))
                            {
                                index--;
                                initial_number_of_variable_inputs++;
                                continue;
                            }

                            // a fixed input, place it at the start of inputs
                            const v = this.inputs.splice(index, 1)[0];
                            this.inputs.splice(0, 0, v);
                        }

                        // force graph redraw to update connections if nothing changed in the next steps
                        this.addInput(`x`, this.input_type);
                        this.removeInput(this.inputs.length-1);

                        console.log(`check: ${initial_number_of_variable_inputs} ; ${new_inputs_len}`)

					    // if there are too many inputs remove them
					    let input_index = initial_total_inputs_len - 1;
					    for(let i= initial_number_of_variable_inputs; i > new_inputs_len; --i)
					        this.removeInput(input_index--);

					    // if there are too few inputs add them
					    for(let i= initial_number_of_variable_inputs; i < new_inputs_len; i++)
                            this.addInput(`${this.var_prefix}${i+1}`, this.input_type);

					},
				}
			);

            return r;
		};

	},
});