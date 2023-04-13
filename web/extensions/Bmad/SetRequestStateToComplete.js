import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.Bmad.SetRequestStateToComplete",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "SetRequestStateToComplete") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: "add input",
							callback: () => {
								const index = this.inputs.length;
								this.addInput("resource_"+index, "TASK_DONE");
							},
						}
						,
						{
							content: "remove all unconnected inputs",
							callback: () => {
								for (let i = this.inputs.length-1; i >= 1; i--) {
									if (!this.inputs[i].link) {
										this.removeInput(i)
									}
								}
								for (let i = this.inputs.length-1; i >= 1; i--) {
									this.inputs[i].name = "resource_" + i;
								}
							},
						}
					);
				}

				return r;
			};
		}
	},
	
});