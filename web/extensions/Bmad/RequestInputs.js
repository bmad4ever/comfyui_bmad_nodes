import { app } from "/scripts/app.js";

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

function SetPropertyValue(widget, node, value) {
    node.properties["values"][widget.name] = value
	node.widgets[1].value = JSON.stringify(node.properties["values"])
}
                  
function RemoveUnnamedOutputs(node){
    for (let i = node.outputs.length-1; i >= 0; i--) 
        if(node.outputs[i].name === "STRING")
            node.removeOutput(i);
}
              
function AddWidgetsForAllProperties(node){
	const type = "STRING";
	for (let i = 0 ; i < node.outputs.length; i++) {
		let name = node.outputs[i].name;
		let value = node.properties[name];
		CreateWidget(node, type, name, value,  function (v, _, node) {SetPropertyValue(this, node, v)});
	}
}

function RemoveWidgetsForAllProperties(node){
	// no output widgets to remove?
	if(node.widgets.length == 2)
		return;
	
	for (let i = node.outputs.length-1; i >= 0; i--) {
		node.widgets.splice(i+2);
	}
	
	node.setSize(node.computeSize());
}

			  
app.registerExtension({
	name: "Comfy.Bmad.RequestInputs",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "RequestInputs") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                RemoveUnnamedOutputs(this);
				if(!this.properties["values"])
					this.properties["values"] = {}
                
				this.getExtraMenuOptions = function(_, options) {                   
					options.unshift(
						{
							content: "add variable",
							callback: () => {
                                const name = this.widgets["0"].value;
                                
                                //check if name is not empty
                                if(name === ""){
                                    alert("Can't create an unnamed output.");
                                    return;
                                }
                                
                                //check if name is different from STRING
                                if(name === "STRING"){
                                    alert("Can't create an output named STRING.");
                                    return;
                                }
                                
                                // check if the input was already added
                                if(name in this.properties["values"]){
                                    alert("Input '" + name +"' already defined. Use a different name.");
                                    return;
                                }
                                
                                const type = "STRING";
								const default_value = "";
                                
								// Add output, property and widget.
								this.addOutput(name, type);
								this.properties["values"][name] = default_value;
                                //CreateWidget(this, type, name, default_value, function (v, _, node) {SetPropertyValue(this, node, v)});
							},
						}
						,
						{
							content: "delete all unconnected",
							callback: () => {
								for (let i = this.outputs.length-1; i >= 0; i--) 
									if (!this.outputs[i].links || this.outputs[i].links.length == 0) {
										delete this.properties["values"][this.outputs[i].name];
                                        if(this.widgets.length > 2)
											this.widgets.splice(i+2);
										this.removeOutput(i);
									}
							},
						}
						,
						{
							content: "show outputs' widgets",
							callback: () => {
								AddWidgetsForAllProperties(this);
							},
						}
						,
						{
							content: "hide outputs' widgets",
							callback: () => {
								RemoveWidgetsForAllProperties(this);
							},
						}
					);
				}

				return r;
			};
		}
	},
	
    

	loadedGraphNode(node, _) {
		if (node.type === "RequestInputs") {
			node.setSize(node.computeSize());
		}
	},
});