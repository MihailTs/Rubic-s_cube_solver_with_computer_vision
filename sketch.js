let topSide = [0,0,0,0,0,0,0,0,0]
let bottomSide = [0,0,0,0,0,0,0,0,0]
let frontSide = [0,0,0,0,0,0,0,0,0]
let leftSide = [0,0,0,0,0,0,0,0,0]
let rightSide = [0,0,0,0,0,0,0,0,0]
let backSide = [0,0,0,0,0,0,0,0,0]

let color_dict = {
    0: [255, 255, 255], //white
    1: [255, 0, 0], // red
    2: [0, 255, 0], // green
    3: [0, 0, 255], // blue
    4: [255, 165, 0], // orange
    5: [255, 255, 0], // yellow
}

let squareSize;
let padding;
let sidePadding;
let cubeSide;
let rotX = 0;
let rotY = 0;

function setup() {
    let cnv = createCanvas(8 * displayWidth / 9, 2 * displayHeight / 3, WEBGL);
    cnv.id("sketch");

    squareSize = width / 40;
    padding = squareSize / 5;
    cubeSide = width / 5;

    // Camera: front-facing, centered at canvas
    let camZ = 2.5 * (height / 2) / tan(PI / 6); // distance from canvas center
    perspective(PI / 6, width / height, 0.1, 10000);
    camera(0, 0, camZ,  // camera position
           0, 0, 0,     // look at origin
           0, 1, 0);    // up vector

    frameRate(30);
}

function draw() {
    background(200);

    drawSide(topCenterCoordinates());
    drawSide(bottomCenterCoordinates());
    drawSide(frontCenterCoordinates());
    drawSide(backCenterCoordinates());
    drawSide(leftCenterCoordinates());
    drawSide(rightCenterCoordinates());

    drawRubiksCube(-width/4, 0, 0, cubeSide);
}

function drawSide(coordinates) {
    let x = coordinates[0];
    let y = coordinates[1];
    square(x - squareSize/2, y - squareSize/2, squareSize, 10);
    square(x - squareSize/2, y - squareSize/2 + squareSize + padding, squareSize, 10);
    square(x - squareSize/2, y - squareSize/2 - squareSize - padding, squareSize, 10);
    square(x - squareSize/2 + padding + squareSize, y - squareSize/2, squareSize, 10);
    square(x - squareSize/2 + padding + squareSize, y - squareSize/2 + squareSize + padding, squareSize, 10);
    square(x - squareSize/2 + padding + squareSize, y - squareSize/2 - squareSize - padding, squareSize, 10);
    square(x - squareSize/2 - padding - squareSize, y - squareSize/2, squareSize, 10);
    square(x - squareSize/2 - padding - squareSize, y - squareSize/2 + squareSize + padding, squareSize, 10);
    square(x - squareSize/2 - padding - squareSize, y - squareSize/2 - squareSize - padding, squareSize, 10);
}

function drawRubiksCube(centerX, centerY, centerZ, size) {
    push();

    // Move to the cube's center
    translate(centerX, centerY, centerZ);
    rotateX(rotX);
    rotateY(rotY);

    let small = size / 3;  // size of each small cube
    let gap = 2;            // small gap between cubes

    // Optional lighting
    ambientLight(150);
    directionalLight(255, 255, 255, 0.5, 1, -1);

    // Loop through 3x3x3 cubes
    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            for (let z = -1; z <= 1; z++) {
                push();
                translate(
                    x * (small + gap),
                    y * (small + gap),
                    z * (small + gap)
                );

                // Example face colors (you can customize per cube or per face)
                let faceColors = {
                    front: color(255, 0, 0),      // red
                    back: color(255, 165, 0),     // orange
                    left: color(0, 0, 255),       // blue
                    right: color(0, 255, 0),      // green
                    top: color(255, 255, 255),    // white
                    bottom: color(255, 255, 0)    // yellow
                };

                drawSmallCubePlanes(small, faceColors);

                pop();
            }
        }
    }

    pop();
}

// Helper function: draws one small cube using 6 planes
function drawSmallCubePlanes(size, faceColors) {
    let half = size / 2;

    push();
    // Front face
    push();
    translate(0, 0, half);
    fill(faceColors.front);
    plane(size, size);
    pop();

    // Back face
    push();
    translate(0, 0, -half);
    rotateY(PI);
    fill(faceColors.back);
    plane(size, size);
    pop();

    // Right face
    push();
    translate(half, 0, 0);
    rotateY(HALF_PI);
    fill(faceColors.right);
    plane(size, size);
    pop();

    // Left face
    push();
    translate(-half, 0, 0);
    rotateY(-HALF_PI);
    fill(faceColors.left);
    plane(size, size);
    pop();

    // Top face
    push();
    translate(0, -half, 0);
    rotateX(HALF_PI);
    fill(faceColors.top);
    plane(size, size);
    pop();

    // Bottom face
    push();
    translate(0, half, 0);
    rotateX(-HALF_PI);
    fill(faceColors.bottom);
    plane(size, size);
    pop();

    pop();
}

function topCenterCoordinates() {
    return [(3 * width / 4) - padding - (3 * squareSize / 2) - sidePadding / 2, 
            (height / 2) - (3 * squareSize) - (2 * padding) - sidePadding];
}

function bottomCenterCoordinates() {
    return [(3 * width / 4) - padding - (3 * squareSize / 2) - sidePadding / 2, 
            (height / 2) + (3 * squareSize) + (2 * padding) + sidePadding];
}

function frontCenterCoordinates() {
    return [(3 * width / 4) - padding - (3 * squareSize / 2) - sidePadding / 2, 
            (height / 2)];
}

function backCenterCoordinates() {
    return [(3 * width / 4) + (3 * padding) + (9 * squareSize / 2) + (3 * sidePadding / 2), 
            (height / 2)];
}

function rightCenterCoordinates() {
    return [(3 * width / 4) + padding + (3 * squareSize / 2) + sidePadding / 2, 
            (height / 2)];
}

function leftCenterCoordinates() {
    return [(3 * width / 4) - (4 * padding) - (9 * squareSize / 2) - (3 * sidePadding / 2), 
            (height / 2)];
}

function getColor(a) {
    return color(color_dict[a][0], color_dict[a][1], color_dict[a][2])
}

function mouseDragged() {
    rotY += movedX * 0.005;
    rotX -= movedY * 0.005;
}