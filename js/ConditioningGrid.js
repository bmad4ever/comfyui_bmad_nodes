import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.Bmad.ConditioningGrid",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!nodeData.name.includes("Conditioning Grid")) return;

		const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {

		    this.grid_type = null
			if(nodeData.name.includes("string"))
				this.grid_type = "STRING"
			else if(nodeData.name.includes("cond"))
				this.grid_type = "CONDITIONING"

		    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;

            // add option
            options.unshift(
				{
					content: "update inputs",
					callback: () => {
					    // remove all conditioning
						for(let i = this.inputs.length-1; i>0; i--)
						    if(this.inputs[i]["type"] === this.grid_type &&  this.inputs[i]["name"] !== "base")
							    this.removeInput(i)

                        const columns = this.widgets.find(w => w.name === "columns")["value"];
                        const rows = this.widgets.find(w => w.name === "rows")["value"];
						for(let j = 1; j <= rows; ++j)
							for(let i = 1; i <= columns; ++i)
								this.addInput(`r${j}_c${i}`, this.grid_type)
					},
				}
			);

            return r;
		};

	},
});