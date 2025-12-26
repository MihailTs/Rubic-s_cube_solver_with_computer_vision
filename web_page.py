from flask import Flask, render_template_string
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Cube Image Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            width: 100%;
            height: 100%;
            gap: 20px;
            padding: 20px;
        }
        
        .left-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(10px);
        }
        
        canvas {
            width: 100%;
            height: 100%;
            display: block;
            border-radius: 10px;
        }
        
        .right-panel {
            flex: 0 0 60%;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
        }
        
        .title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        
        .content-wrapper {
            display: flex;
            gap: 12px;
            flex: 1;
        }
        
        .unfolded-cube {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            flex: 1;
        }
        
        .cube-face {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 3px;
            backdrop-filter: blur(10px);
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2px;
        }
        
        .cube-face.top { grid-column: 2; grid-row: 1; }
        .cube-face.front { grid-column: 2; grid-row: 2; }
        .cube-face.right { grid-column: 3; grid-row: 2; }
        .cube-face.left { grid-column: 1; grid-row: 2; }
        .cube-face.back { grid-column: 4; grid-row: 2; }
        .cube-face.bottom { grid-column: 2; grid-row: 3; }
        
        .face-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 12px;
            font-weight: bold;
            position: absolute;
            margin: 2px;
        }
        
        .cube-face {
            position: relative;
        }
        
        .small-cube {
            aspect-ratio: 1;
            border: 1px solid rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .small-cube:not(.center) {
            cursor: pointer;
        }
        
        .small-cube:not(.center):hover {
            transform: scale(1.05);
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.5);
        }
        
        .small-cube.center {
            cursor: not-allowed;
        }
        
        .info-box {
            background: rgba(255, 255, 255, 0.15);
            color: white;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            font-size: 12px;
            line-height: 1.5;
            backdrop-filter: blur(10px);
        }
        
        .button-group {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            min-width: 120px;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: 2px solid white;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        input[type="file"] {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <canvas id="canvas"></canvas>
        </div>
        
        <div class="right-panel">
            <div class="title">Cube Unfolding</div>
            
            <div class="content-wrapper">
                <div class="unfolded-cube" id="unfoldedCube"></div>
                
                <div class="button-group">
                    <button class="upload-btn" onclick="document.getElementById('file1').click()">📁 Upload 1</button>
                    <button class="upload-btn" onclick="document.getElementById('file2').click()">📁 Upload 2</button>
                    <button class="upload-btn" onclick="document.getElementById('file3').click()">📁 Upload 3</button>
                    <button class="upload-btn" onclick="document.getElementById('file4').click()">📁 Upload 4</button>
                    <button class="upload-btn" onclick="document.getElementById('file5').click()">📁 Upload 5</button>
                    <button class="upload-btn" onclick="document.getElementById('file6').click()">📁 Upload 6</button>
                </div>
            </div>
            
            <input type="file" id="file1" accept="image/*">
            <input type="file" id="file2" accept="image/*">
            <input type="file" id="file3" accept="image/*">
            <input type="file" id="file4" accept="image/*">
            <input type="file" id="file5" accept="image/*">
            <input type="file" id="file6" accept="image/*">
            
            <div class="info-box">
                <strong>Click on the 3D cube or unfolding squares to change colors</strong><br>
                Center cubes are unclickable. Colors cycle through: White → Red → Green → Blue → Yellow → Orange
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const canvas = document.getElementById('canvas');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setClearColor(0x000000, 0.1);
        camera.position.z = 3;
        
        const cubeGroup = new THREE.Group();
        scene.add(cubeGroup);
        
        // Colors for cycling: white + 5 colors
        const cycleColors = [0xFFFFFF, 0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF8C00];
        
        // Face-specific colors for center cubes: [front, top, left, bottom, right, back]
        const faceColors = {
            0: 0xFFFFFF,  // front - white
            1: 0xFF0000,  // top - red
            2: 0x00FF00,  // left - green
            3: 0x0000FF,  // bottom - blue
            4: 0xFFFF00,  // right - yellow
            5: 0xFF8C00   // back - orange
        };

        const topSide = [];
        const frontSide = [];
        const leftSide = [];
        const rightSide = [];
        const bottomSide = [];
        const backSide = [];
        for(let i = 0; i < 9; i++) {
            topSide.push(0);
            frontSide.push(0);
            leftSide.push(0);
            rightSide.push(0);
            bottomSide.push(0);
            backSide.push(0);
        }
        
        topSide[4] = 2;
        frontSide[4] = 4;
        bottomSide[4] = 3;
        leftSide[4] = 1;
        rightSide[4] = 0;
        backSide[4] = 5;
        
        const cubeData = [];
        const cubes = [];
        const boxSize = 0.5;
        const spacing = 0.51;
        
        let cubeIndex = 0;
        
        function isCenterOfFace(x, y, z) {
            if (x === 1 && y === 0 && z === 0) return 0; // right center
            if (x === -1 && y === 0 && z === 0) return 1; // left center
            if (x === 0 && y === 1 && z === 0) return 2; // top center
            if (x === 0 && y === -1 && z === 0) return 3; // bottom center
            if (x === 0 && y === 0 && z === 1) return 4; // front center
            if (x === 0 && y === 0 && z === -1) return 5; // back center
            return -1;
        }
        
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                for (let z = -1; z <= 1; z++) {
                    const geometry = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
                    
                    const materials = [];
                    const centerFaceIndex = isCenterOfFace(x, y, z);
                    const isCenter = (x === 0 && y === 0 && z === 0);
                    
                    // Create materials for all 6 faces
                    for (let i = 0; i < 6; i++) {
                        let color = 0xFFFFFF; // default white
                        
                        // If this is a center cube of a face, use the face color
                        if (centerFaceIndex >= 0 && i === centerFaceIndex) {
                            color = faceColors[i];
                        }
                        
                        materials.push(new THREE.MeshPhongMaterial({ color: color }));
                    }

                    const cube = new THREE.Mesh(geometry, materials);
                    cube.position.set(x * spacing, y * spacing, z * spacing);
                    cube.userData.cubeIndex = cubeIndex;
                    cube.userData.faceColors = [0, 0, 0, 0, 0, 0]; // Track color index for each face
                    
                    // Add black edges
                    const edges = new THREE.EdgesGeometry(geometry);
                    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });
                    const wireframe = new THREE.LineSegments(edges, lineMaterial);
                    cube.add(wireframe);
                    
                    cubeGroup.add(cube);
                    cubes.push(cube);
                    
                    cubeData.push({
                        index: cubeIndex,
                        x, y, z,
                        mesh: cube,
                    });
                    cubeIndex++;
                }
            }
        }
        
        // Lighting
        const light1 = new THREE.DirectionalLight(0xffffff, 1);
        light1.position.set(5, 5, 5);
        scene.add(light1);
        
        const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
        light2.position.set(-5, -5, 5);
        scene.add(light2);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        // Raycaster for clicking
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        // Mouse controls
        let mouseDown = false;
        let mouseX = 0, mouseY = 0;
        let rotationX = 0, rotationY = 0;
        let dragDistance = 0;
        
        canvas.addEventListener('mousedown', (e) => {
            mouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
            dragDistance = 0;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (mouseDown) {
                const dx = e.clientX - mouseX;
                const dy = e.clientY - mouseY;
                dragDistance += Math.abs(dx) + Math.abs(dy);
                rotationY += dx * 0.01;
                rotationX += dy * 0.01;
                mouseX = e.clientX;
                mouseY = e.clientY;
            }
        });
        
        canvas.addEventListener('mouseup', (e) => {
            if (dragDistance < 5) {
                const rect = canvas.getBoundingClientRect();
                mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(cubes);
                
                if (intersects.length > 0) {
                    const hit = intersects[0];
                    const mesh = hit.object;
                    const faceIndex = hit.face.materialIndex;
                    
                    // Check if clicking on center cube - they are unclickable
                    if (mesh.userData.cubeIndex == 4 || 
                        mesh.userData.cubeIndex == 10 ||
                        mesh.userData.cubeIndex == 12 ||
                        mesh.userData.cubeIndex == 14 ||
                        mesh.userData.cubeIndex == 16 ||
                        mesh.userData.cubeIndex == 22) {
                        return;
                    }
                    
                    // Cycle through colors for other cubes
                    mesh.userData.faceColors[faceIndex] = (mesh.userData.faceColors[faceIndex] + 1) % cycleColors.length;
                    mesh.material[faceIndex].color.setHex(cycleColors[mesh.userData.faceColors[faceIndex]]);
                    updateAllUnfolding();
                }
            }
            mouseDown = false;
        });
        
        canvas.addEventListener('mouseleave', () => {
            mouseDown = false;
        });
        
        // Create unfolding visualization
        const faceNames = ['right', 'left', 'top', 'bottom', 'front', 'back'];
        const unfoldedCube = document.getElementById('unfoldedCube');
        
        function createUnfolding() {
            unfoldedCube.innerHTML = '';
            
            faceNames.forEach((faceName, faceIndex) => {
                const faceDiv = document.createElement('div');
                faceDiv.className = `cube-face ${faceName}`;
                faceDiv.dataset.faceIndex = faceIndex;
                
                const label = document.createElement('div');
                label.className = 'face-label';
                label.textContent = faceName.toUpperCase();
                faceDiv.appendChild(label);
                
                // Create 3x3 grid for each face
                for (let i = 0; i < 9; i++) {
                    const smallCube = document.createElement('div');
                    smallCube.className = 'small-cube';
                    smallCube.dataset.faceIndex = faceIndex;
                    smallCube.dataset.position = i;
                    
                    // Position 4 is the center
                    if (i === 4) {
                        smallCube.classList.add('center');
                    } else {
                        // Add click listener to non-center squares
                        smallCube.addEventListener('click', handleUnfoldingClick);
                    }
                    
                    faceDiv.appendChild(smallCube);
                }
                
                unfoldedCube.appendChild(faceDiv);
            });
            
            updateAllUnfolding();
        }

        function faceRowColToXYZ(face, row, col) {
            // row, col ∈ {0,1,2} → coord ∈ {-1,0,1}
            const a = col - 1;
            const b = 1 - row;

            switch (face) {
                case 0: return { x:  1, y:  b, z: -a }; // RIGHT
                case 1: return { x: -1, y:  b, z:  a }; // LEFT
                case 2: return { x:  a, y:  1, z: -b }; // TOP
                case 3: return { x:  a, y: -1, z:  b }; // BOTTOM
                case 4: return { x:  a, y:  b, z:  1 }; // FRONT
                case 5: return { x: -a, y:  b, z: -1 }; // BACK
            }
        }
        
        function handleUnfoldingClick(e) {
            const faceIndex = parseInt(e.target.dataset.faceIndex);
            const position = parseInt(e.target.dataset.position);

            const row = Math.floor(position / 3);
            const col = position % 3;

            const target = faceRowColToXYZ(faceIndex, row, col);

            cubes.forEach(cube => {
                const x = Math.round(cube.position.x / spacing);
                const y = Math.round(cube.position.y / spacing);
                const z = Math.round(cube.position.z / spacing);

                if (x === target.x && y === target.y && z === target.z) {
                    if ([4,10,12,14,16,22].includes(cube.userData.cubeIndex)) return;

                    cube.userData.faceColors[faceIndex] =
                        (cube.userData.faceColors[faceIndex] + 1) % cycleColors.length;

                    cube.material[faceIndex].color
                        .setHex(cycleColors[cube.userData.faceColors[faceIndex]]);

                    updateAllUnfolding();
                }
            });
        }



        function updateAllUnfolding() {
            const smallCubes = document.querySelectorAll('.small-cube');
            
            smallCubes.forEach(smallCube => {
                const faceIndex = parseInt(smallCube.dataset.faceIndex);
                const position = parseInt(smallCube.dataset.position);
                
                if (position === 4) {
                    // Center square - colored based on face
                    const baseHex = faceColors[faceIndex].toString(16).padStart(6, '0');
                    smallCube.style.backgroundColor = '#' + baseHex;
                } else {
                    // Find matching cube and get its color
                    let foundColor = cycleColors[0];

                    const row = Math.floor(position / 3);
                    const col = position % 3;
                    const target = faceRowColToXYZ(faceIndex, row, col);

                    cubes.forEach(cube => {
                        const x = Math.round(cube.position.x / spacing);
                        const y = Math.round(cube.position.y / spacing);
                        const z = Math.round(cube.position.z / spacing);

                        if (x === target.x && y === target.y && z === target.z) {
                            foundColor = cycleColors[cube.userData.faceColors[faceIndex]];
                        }
                    });

                    const colorHex = foundColor.toString(16).padStart(6, '0');
                    smallCube.style.backgroundColor = '#' + colorHex;
                }
            });
        }
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            cubeGroup.rotation.x = rotationX;
            cubeGroup.rotation.y = rotationY;
            
            renderer.render(scene, camera);
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        });
        
        createUnfolding();
        animate();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)