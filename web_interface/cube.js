let s1 = (sketch) => {
    let rotX = 0;
    let rotY = 0;
    let cubeSide;

    sketch.setup = function() {
        let cnv = sketch.createCanvas(2 * sketch.windowWidth / 5, 3 * sketch.windowHeight / 4, sketch.WEBGL);
        cnv.parent('cubeContainer');
        cubeSide = sketch.width / 2.8;
        let camZ = 2.5 * (sketch.height / 2) / sketch.tan(sketch.PI / 6);
        sketch.perspective(sketch.PI / 6, sketch.width / sketch.height, 0.1, 10000);
        sketch.camera(0, 0, camZ, 0, 0, 0, 0, 1, 0);
        sketch.frameRate(10);
    }

    sketch.draw = function() {
        sketch.background(200);
        drawRubiksCube(0, 0, 0, cubeSide);
    }

    function drawRubiksCube(centerX, centerY, centerZ, size) {
        sketch.push();
        sketch.translate(centerX, centerY, centerZ);
        sketch.rotateX(rotX);
        sketch.rotateY(rotY);
        let small = size / 3;
        let gap = 2;
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                for (let z = -1; z <= 1; z++) {
                    sketch.push();
                    sketch.translate(x * (small + gap), y * (small + gap), z * (small + gap));
                    let faceColors = {
                        front: sketch.color(255, 0, 0),
                        back: sketch.color(255, 165, 0),
                        left: sketch.color(0, 0, 255),
                        right: sketch.color(0, 255, 0),
                        top: sketch.color(255, 255, 255),
                        bottom: sketch.color(255, 255, 0)
                    };
                    drawSmallCubePlanes(small, faceColors);
                    sketch.pop();
                }
            }
        }
        sketch.pop();
    }

    function drawSmallCubePlanes(size, faceColors) {
        let half = size / 2;
        sketch.push();
        sketch.push();
        sketch.translate(0, 0, half);
        sketch.fill(faceColors.front);
        sketch.plane(size, size);
        sketch.pop();
        sketch.push();
        sketch.translate(0, 0, -half);
        sketch.rotateY(sketch.PI);
        sketch.fill(faceColors.back);
        sketch.plane(size, size);
        sketch.pop();
        sketch.push();
        sketch.translate(half, 0, 0);
        sketch.rotateY(sketch.HALF_PI);
        sketch.fill(faceColors.right);
        sketch.plane(size, size);
        sketch.pop();
        sketch.push();
        sketch.translate(-half, 0, 0);
        sketch.rotateY(-sketch.HALF_PI);
        sketch.fill(faceColors.left);
        sketch.plane(size, size);
        sketch.pop();
        sketch.push();
        sketch.translate(0, -half, 0);
        sketch.rotateX(sketch.HALF_PI);
        sketch.fill(faceColors.top);
        sketch.plane(size, size);
        sketch.pop();
        sketch.push();
        sketch.translate(0, half, 0);
        sketch.rotateX(-sketch.HALF_PI);
        sketch.fill(faceColors.bottom);
        sketch.plane(size, size);
        sketch.pop();
        sketch.pop();
    }

    sketch.mouseDragged = function() {
        if(sketch.mouseX < sketch.width) {
            rotY += sketch.movedX * 0.005;
            rotX -= sketch.movedY * 0.005;
        }
    }
};

new p5(s1);