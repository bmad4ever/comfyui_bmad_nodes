import { ComfyWidgets, addValueControlWidget } from "../../scripts/widgets.js";
import { app } from "/scripts/app.js";

var mo_node_types = {
    "FromListGetMasks":  "MASK",
    "FromListGetImages":  "IMAGE",
    "FromListGetLatents":  "LATENT",
    "FromListGetConds":  "CONDITIONING",
    "FromListGetModels": "MODEL",
    "FromListGetColors": "COLOR",
    "FromListGetStrings": "STRING",
    "FromListGetInts": "INT",
    "FromListGetFloats": "FLOAT",
}


function RemoveAllOutputs(node){
    for (let i = node.outputs.length-1; i >= 0; i--)
            node.removeOutput(i);
}

/**
* @description
* Adds or removes output slots according to the number in the "outputs" widget
*/
function UpdateOutputs(node){
    const new_outputs_len = node.widgets.find(w => w.name === "outputs")["value"];
    const current_outputs_len = node.inputs === undefined? 0 : node.outputs.length;

    for(let i= current_outputs_len; i > new_outputs_len; --i)
        node.removeOutput(i-1);

    for(let i= current_outputs_len; i < new_outputs_len; i++)
        node.addOutput(`${node.var_prefix}${i+1}`, node.output_type);
}

app.registerExtension({
	name: "Comfy.Bmad.ArbitraryOutputsFixedInputs",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if( !(nodeData.name in mo_node_types) ) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
				const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
				RemoveAllOutputs(this)
				ComfyWidgets.INT(this, "outputs", ["INT", {default:0, min:0, max:16}], app)
                this.output_type = mo_node_types[nodeData.name];
                this.var_prefix = this.output_type.toLowerCase() + "_"
				return r;
		};

		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {
		    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;

            // add option
            options.unshift(
				{
					content: "update Outputs",
					callback: () => {UpdateOutputs(this)},
				}
			);

            return r;
		};

	},
});