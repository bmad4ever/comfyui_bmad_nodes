import { app } from "/scripts/app.js";


const node_name = "AnyToAny"
const wild_type_name = "*"

// copy paste, maybe should create another file to avoid repeating myself
function RemoveUnnamedOutputs(node){
    for (let i = node.outputs.length-1; i >= 0; i--)
        if(node.outputs[i].name === wild_type_name)
            node.removeOutput(i);
}

function CreateWidget(node, inputType, inputName, val, func, config = {}) {
    return {
        widget: node.addWidget(
            inputType,
            inputName,
            val,
            func,
            config
        ),
    };
}

function UpdateSizes(inputs_delta, node){
// I am a bit confused on the details of how exactly this works...
// but it seems to be kinda working with this, so I will leave as it is for now...
    const input_unit_size = 20;
    const size_diff = input_unit_size * inputs_delta;

    let w0Height = node.inputHeight; // text rect height apparently

    let size = node.computeSize();
    size[1] += size_diff;
    node.setSize(size);

    node.widgets[0].y += size_diff ;
    if(inputs_delta < 0) node.widgets[1].y += size_diff ;
    node.inputHeight = w0Height;
}

app.registerExtension({
	name: `Comfy.Bmad.{node_name}`,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		if (nodeData.name !== node_name) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            RemoveUnnamedOutputs(this);
            CreateWidget(this , "string", "New Variable Name", "any", function (v, _, node) {this.value = v.trim();})

			this.getExtraMenuOptions = function(_, options) {
				options.unshift(
					{
						content: "add variable",
						callback: () => {
                               this.widgets["1"].value = this.widgets["1"].value.trim();
                               const name = this.widgets["1"].value;

                               //check if name is not empty
                               if(name === ""){
                                   alert("Can't create an unnamed output.");
                                   return;
                               }
                               // check if the output was already added
                               if(this.outputs !== 'undefined')
                                   for (let i = this.outputs.length-1; i >= 0; i--)
                                       if(name === this.outputs[i].name){
                                           alert("Output '" + name +"' already defined. Use a different name.");
                                           return;
                                       }
							this.addOutput(name, wild_type_name);

                            if(this.outputs.length < 2) return;
                            UpdateSizes(+1, this)
						},
					}
					,
					{
						content: "delete variable",
						callback: () => {
						    this.widgets["0"].value = this.widgets["0"].value.trim();
                            const name = this.widgets["1"].value;

                            for (let i = this.outputs.length-1; i >= 0; i--)
								if (this.outputs[i].name === name){
									this.removeOutput(i);
									if(this.outputs.length >= 1) UpdateSizes(-1, this)
									return;
								}
						},
					},
					{
						content: "delete all unconnected",
						callback: () => {
						    let init_len = this.outputs.length;
							for (let i = this.outputs.length-1; i >= 0; i--)
								if (!this.outputs[i].links || this.outputs[i].links.length == 0)
									    this.removeOutput(i);

								if(init_len < 2) return;
								let removed = init_len - this.outputs.length;
								if(this.outputs.length < 1) removed -= (1-this.outputs.length)
								UpdateSizes(-removed, this)
						},
					}
				);
			}
			return r;
		};
	},



	loadedGraphNode(node, _) {
		if (node.type === node_name) {
			node.setSize(node.computeSize());
		}
	},
});