import { app } from "/scripts/app.js";


const number_of_nonoutput_widgets = 2;

function NumberOfOutputWidgets(node){
    return node.widgets.length - number_of_nonoutput_widgets;
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

function SetPropertyValue(name, node, value) {
    node.properties["values"][name] = value
	node.widgets[1].value = JSON.stringify(node.properties["values"])
}
                  
function RemoveUnnamedOutputs(node){
    for (let i = node.outputs.length-1; i >= 0; i--) 
        if(node.outputs[i].name === "STRING")
            node.removeOutput(i);
}
              
function AddWidgetsForAllProperties(node){
    let number_of_output_widgets = NumberOfOutputWidgets(node);
    
    // all widgets already displayed? if so, no need to add.
	if(number_of_output_widgets == node.outputs.length)
		return;
	
	const type = "STRING";
	for (let i = number_of_output_widgets; i < node.outputs.length; i++) {
		let name = node.outputs[i].name;
		let value = node.properties["values"][name];
		CreateWidget(node, type, name, value,  function (v, _, node) {SetPropertyValue(this.name, node, v)});
	}
}

function RemoveWidgetsForAllProperties(node){
    let number_of_output_widgets = NumberOfOutputWidgets(node);
    
	// no output widgets to remove?
	if(number_of_output_widgets == 0)
		return;
	
	for (let i = number_of_output_widgets - 1; i >= 0; i--) {
		node.widgets.splice(i + number_of_nonoutput_widgets);
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
					this.properties["values"] = {};
				
				CreateWidget(this , "string", "New Variable Name", "image", function (v, _, node) {this.value = v.trim();})
				
				// set values as second widget, also works as a separator of sorts
				const widget_swap = this.widgets[0]
				this.widgets[0] = this.widgets[1]
				this.widgets[1] = widget_swap
				
				this.widgets[1]["disabled"] = true;

                
				this.getExtraMenuOptions = function(_, options) {                   
					options.unshift(
						{
							content: "add variable",
							callback: () => {
                                this.widgets["0"].value = this.widgets["0"].value.trim();
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
                                SetPropertyValue(name, this, default_value);
                                
                                //if showing output widgets, show the new variable
                                if(NumberOfOutputWidgets(this) > 0) 
                                    AddWidgetsForAllProperties(this);
							},
						}
						,
						{
							content: "delete all unconnected",
							callback: () => {
								for (let i = this.outputs.length-1; i >= 0; i--) 
									if (!this.outputs[i].links || this.outputs[i].links.length == 0) {
										delete this.properties["values"][this.outputs[i].name];
                                        if(this.widgets.length > number_of_nonoutput_widgets && i < this.widgets.length)
											this.widgets.splice(i + number_of_nonoutput_widgets, 1);
										this.removeOutput(i);
									}
								//update values
								this.widgets[1].value = JSON.stringify(this.properties["values"])
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