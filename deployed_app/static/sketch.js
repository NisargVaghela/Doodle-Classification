let rdpPoints = []
let drawingPoints = []
let allPoints = []
let st=[]
let x=[]
let y=[]
let epsilon = 10
let isDrawing = false;

function setup() {
	createCanvas(1400,800);
	background(255);	
}

function draw() {
 strokeWeight(10);
 stroke(0); 
}


function mouseDragged() {  
  isDrawing = true;
  allPoints.push(createVector(mouseX,mouseY))
  line(pmouseX, pmouseY, mouseX, mouseY);
}

function mouseReleased(){
  if(isDrawing){

	    let total = allPoints.length;
	    let start = allPoints[0];
	    let end = allPoints[total - 1];
	    rdpPoints.push(start);
	    rdp(0, total - 1, allPoints, rdpPoints);
	    rdpPoints.push(end);
	    drawingPoints.push(rdpPoints)
	    
	    x=[]
	    y=[]
	    for(let p of rdpPoints){
	    	x.push(p.x)
	    	y.push(p.y)
	    }
	    st.push([x,y])
	  	console.log(st)

	  	clr();
	    stroke(0);  
	    for (var i = 0; i < drawingPoints.length; i++){    
	      for (var j = 1; j < drawingPoints[i].length; j++) {
	      	line(drawingPoints[i][j-1].x,drawingPoints[i][j-1].y,drawingPoints[i][j].x,drawingPoints[i][j].y)
	      }
	    }

	  	json_data = {"data" : st}

	  	$.ajax({
  				type: "POST",
  				contentType: 'application/json; charset=utf-8',
  				url: '/predict',
  				data: JSON.stringify(json_data),
  				async: false,
  				success: function(response) {  						
  					if (response['ans']) {
  						console.log(response['ans'])
              sneakbar(response['ans'])
  					}else{
  						console.log("couldn't understand your drawing!")
              sneakbar("couldn't understand your drawing!")
  					}
  				},
  				dataType: "json"
			});

	    	  
	    x=[]
	    y=[]
	    allPoints=[]
	    rdpPoints=[]	    
	}
}


function clr(){
	background(255);  
	isDrawing = false;	
}

function reset() {
	background(255);
	drawingPoints = [];
  st = [];
}

function sneakbar(msg) {
  var x = document.getElementById("snackbar");
  x.innerHTML = msg;
  x.className = "show";
  setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
}

function rdp(startIndex, endIndex, allPoints, rdpPoints) {
  const nextIndex = findFurthest(allPoints, startIndex, endIndex);
  if (nextIndex > 0) {
    if (startIndex != nextIndex) {
      rdp(startIndex, nextIndex, allPoints, rdpPoints);
    }
    rdpPoints.push(allPoints[nextIndex]);
    if (endIndex != nextIndex) {
      rdp(nextIndex, endIndex, allPoints, rdpPoints);
    }
  }
}


function findFurthest(points, a, b) {
  let recordDistance = -1;
  const start = points[a];
  const end = points[b];
  let furthestIndex = -1;
  for (let i = a + 1; i < b; i++) {
    const currentPoint = points[i];
    const d = lineDist(currentPoint, start, end);
    if (d > recordDistance) {
      recordDistance = d;
      furthestIndex = i;
    }
  }
  if (recordDistance > epsilon) {
    return furthestIndex;
  } else {
    return -1;
  }
}


function lineDist(c, a, b) {
  const norm = scalarProjection(c, a, b);
  return p5.Vector.dist(c, norm);
}

function scalarProjection(p, a, b) {
  const ap = p5.Vector.sub(p, a);
  const ab = p5.Vector.sub(b, a);
  ab.normalize(); // Normalize the line
  ab.mult(ap.dot(ab));
  const normalPoint = p5.Vector.add(a, ab);
  return normalPoint;
}