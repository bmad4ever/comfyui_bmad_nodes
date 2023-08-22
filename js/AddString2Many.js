import { app } from "/scripts/app.js";

function RemoveUnnamedOutputs(node){
    for (let i = node.outputs.length-1; i >= 0; i--)
        if(node.outputs[i].name === "STRING")
            node.removeOutput(i);
}

app.registerExtension({
	name: "Comfy.Bmad.AddString2Many",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		if (nodeData.name !== "Add String To Many") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
				const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
				RemoveUnnamedOutputs(this);
                this.input_type = "STRING";
		        this.output_type = "STRING";
				return r;
		};

		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
		    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;

            // add option
            options.unshift(
				{
					content: "update I/Os",
					callback: () => {
    					const new_inputs_len = this.widgets.find(w => w.name === "inputs_len")["value"];
					    const initial_total_inputs_len = this.inputs === undefined? 0 : this.inputs.length;
					    const initial_outputs_len = this.inputs === undefined? 0 : this.outputs.length;
					    let initial_inputs_len = initial_total_inputs_len; // the number of inputs ignoring 'to_add'

					    // if to_add is an input, take note and place it at the top of the inputs
					    for(let i = 0 ; i<initial_total_inputs_len; ++i)
					    {
					        if(this.inputs[i]["name"] === "to_add")
					        {
					            initial_inputs_len = initial_total_inputs_len - 1;

					            if(i>0)
					            {
					                // move to the top of the node
					                const to_add = this.inputs.splice(i, 1)[0];
                                    console.log(to_add);
                                    this.inputs.splice(0, 0, to_add);

                                    // force graph redraw to update connections, in case nothing changed.
                                    this.addInput(`x`, this.input_type);
                                    this.removeInput(this.inputs.length-1);
                                }

					            break;
					        }
					    }

					    // if there are too many inputs remove them
					    // note that: 'to_add', if input, is at index 0, so there is no need to check for it.
					    let input_index = initial_total_inputs_len - 1;
					    for(let i= initial_inputs_len; i > new_inputs_len; --i)
					        this.removeInput(input_index--);

					    // if there are too few inputs add them
					    for(let i= initial_inputs_len; i < new_inputs_len; i++)
                            this.addInput(`i${i+1}`, this.input_type);

					    // repeat previous 2 steps for the outputs
					    for(let i= initial_outputs_len; i > new_inputs_len; --i)
					        this.removeOutput(i-1);

					    for(let i= initial_outputs_len; i < new_inputs_len; i++)
                            this.addOutput(`o${i+1}`, this.output_type);
					},
				}
			);

            return r;
		};

	},
});