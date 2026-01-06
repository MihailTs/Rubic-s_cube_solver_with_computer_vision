let s2 = (sketch) => {
    let color_dict = {
        0: [255, 255, 255],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 165, 0],
        5: [255, 255, 0],
    }

    let topSide = [0,0,0,0,0,0,0,0,0]
    let bottomSide = [0,0,0,0,5,0,0,0,0]
    let frontSide = [0,0,0,0,2,0,0,0,0]
    let leftSide = [0,0,0,0,4,0,0,0,0]
    let rightSide = [0,0,0,0,1,0,0,0,0]
    let backSide = [0,0,0,0,3,0,0,0,0]

    let squareSize;
    let padding;
    let sidePadding;

    sketch.setup = function() {
        let cnv = sketch.createCanvas(2 * sketch.windowWidth / 5, 3 * sketch.windowHeight / 4);
        cnv.parent('unfoldContainer');
        squareSize = sketch.width / 19;
        padding = squareSize / 5;
        sidePadding = 3 * squareSize / 2;
        sketch.frameRate(15);
    }

    sketch.draw = function() {
        sketch.background(200);
        drawSide(topCenterCoordinates(), topSide);
        drawSide(bottomCenterCoordinates(), bottomSide);
        drawSide(frontCenterCoordinates(), frontSide);
        drawSide(backCenterCoordinates(), backSide);
        drawSide(leftCenterCoordinates(), leftSide);
        drawSide(rightCenterCoordinates(), rightSide);
    }

    function drawSide(coordinates, sideColors) {
        let x = coordinates[0];
        let y = coordinates[1];
        sketch.fill(getColor(sideColors[4]));
        sketch.square(x - squareSize/2, y - squareSize/2, squareSize, 5);
        sketch.fill(getColor(sideColors[7]));
        sketch.square(x - squareSize/2, y - squareSize/2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[1]));
        sketch.square(x - squareSize/2, y - squareSize/2 - squareSize - padding, squareSize, 5);
        sketch.fill(getColor(sideColors[5]));
        sketch.square(x - squareSize/2 + padding + squareSize, y - squareSize/2, squareSize, 5);
        sketch.fill(getColor(sideColors[8]));
        sketch.square(x - squareSize/2 + padding + squareSize, y - squareSize/2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[2]));
        sketch.square(x - squareSize/2 + padding + squareSize, y - squareSize/2 - squareSize - padding, squareSize, 5);
        sketch.fill(getColor(sideColors[3]));
        sketch.square(x - squareSize/2 - padding - squareSize, y - squareSize/2, squareSize, 5);
        sketch.fill(getColor(sideColors[6]));
        sketch.square(x - squareSize/2 - padding - squareSize, y - squareSize/2 + squareSize + padding, squareSize, 5);
        sketch.fill(getColor(sideColors[0]));
        sketch.square(x - squareSize/2 - padding - squareSize, y - squareSize/2 - squareSize - padding, squareSize, 5);
    }

    function topCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2, 
                sketch.height / 2 - (3 * squareSize) - (2 * padding) - sidePadding];
    }

    function bottomCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2, 
                sketch.height / 2  + (3 * squareSize) + (2 * padding) + sidePadding];
    }

    function frontCenterCoordinates() {
        return [(sketch.width / 2) - padding - (3 * squareSize / 2) - sidePadding / 2, 
                sketch.height / 2];
    }

    function backCenterCoordinates() {
        return [(sketch.width / 2) + (3 * padding) + (9 * squareSize / 2) + (3 * sidePadding / 2), 
                sketch.height / 2];
    }

    function rightCenterCoordinates() {
        return [(sketch.width / 2) + padding + (3 * squareSize / 2) + sidePadding / 2, 
                sketch.height / 2];
    }

    function leftCenterCoordinates() {
        return [(sketch.width / 2) - (3 * padding) - (9 * squareSize / 2) - (3 * sidePadding / 2), 
                sketch.height / 2];
    }

    function getColor(a) {
        return sketch.color(color_dict[a][0], color_dict[a][1], color_dict[a][2])
    }

    function checkSideClicked(centerCoordinates, checkSide) {
        // centers remain unchangable
        // if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 && sketch.mouseX < centerCoordinates[0] + squareSize / 2 && sketch.mouseY > centerCoordinates[1] - squareSize / 2 && sketch.mouseY < centerCoordinates[1] + squareSize / 2) {
        //     checkSide[4] = (checkSide[4] + 1) % 6;
        if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 && sketch.mouseX < centerCoordinates[0] + squareSize / 2 && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[1] = (checkSide[1] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 && sketch.mouseX < centerCoordinates[0] + squareSize / 2 && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[7] = (checkSide[7] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 && sketch.mouseY < centerCoordinates[1] + squareSize / 2) {
            checkSide[3] = (checkSide[3] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 && sketch.mouseY < centerCoordinates[1] + squareSize / 2) {
            checkSide[5] = (checkSide[5] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[0] = (checkSide[0] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] - squareSize / 2 - squareSize - padding && sketch.mouseY < centerCoordinates[1] - squareSize / 2 - padding) {
            checkSide[2] = (checkSide[2] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] - squareSize / 2 - squareSize - padding && sketch.mouseX < centerCoordinates[0] - squareSize / 2 - padding && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[6] = (checkSide[6] + 1) % 6;
        } else if (sketch.mouseX > centerCoordinates[0] + squareSize / 2 + padding && sketch.mouseX < centerCoordinates[0] + squareSize / 2 + squareSize + padding && sketch.mouseY > centerCoordinates[1] + squareSize / 2 + padding && sketch.mouseY < centerCoordinates[1] + squareSize / 2 + squareSize + padding) {
            checkSide[8] = (checkSide[8] + 1) % 6;
        }
    }

    sketch.mouseClicked = function() {
        checkSideClicked(frontCenterCoordinates(), frontSide);
        checkSideClicked(topCenterCoordinates(), topSide);
        checkSideClicked(leftCenterCoordinates(), leftSide);
        checkSideClicked(rightCenterCoordinates(), rightSide);
        checkSideClicked(bottomCenterCoordinates(), bottomSide);
        checkSideClicked(backCenterCoordinates(), backSide);
    }
};

new p5(s2);