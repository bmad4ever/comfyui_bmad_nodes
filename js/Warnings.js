import { app } from "/scripts/app.js";

var potentially_dangerous_nodes = [
    "BuildColorRangeAdvanced (hsv)",
	"Filter Contour",
	"AnyToAny"
]

var warned = false

app.registerExtension({
	name: "Comfy.Bmad.Warnings",
	loadedGraphNode(node, _) {
		
		 // this will also execute when using dirty-undo, 
		 //  so after 1st warning I just paint the node instead a sendeng the warning again
		
		if (potentially_dangerous_nodes.indexOf(node.type) > -1) {
			if(! warned){
				alert(`The loaded workflow contains potentially DANGEROUS nodes.\n\n`+
				"If you are unsure about the workflow origin check the WHITE PAINTED node(s) before executing.\n\n"+
				"This warning will not reapper.")
				warned = true;
			}
			node.bgcolor = "#FFF"
		}
	},
})




