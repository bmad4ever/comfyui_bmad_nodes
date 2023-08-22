import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.Bmad.ConditioningCombineMultiple",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!nodeData.name.includes("Conditioning (combine ")) return;

		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {

            this.inputs_offset = nodeData.name.includes("selective")?1:0
			this.cond_type = "CONDITIONING"
		    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;

            // add option
            options.unshift(
				{
					content: "update inputs",
					callback: () => {
					    if(!this.inputs)
					        this.inputs=[]

					    const target_number_of_inputs = this.widgets.find(w => w.name === "combine")["value"];
					    if(target_number_of_inputs===this.inputs.length)return; // already set, do nothing

					    if(target_number_of_inputs < this.inputs.length){
    						for(let i = this.inputs.length; i>=this.inputs_offset+target_number_of_inputs; i--)
	    					    //if(this.inputs[i]["type"] === this.cond_type) this should always be true
							      this.removeInput(i)
					    }
                        else{
						    for(let i = this.inputs.length+1-this.inputs_offset; i <= target_number_of_inputs; ++i)
						    	this.addInput(`c${i}`, this.cond_type)
                        }
					},
				}
			);

            return r;
		};

	},
});