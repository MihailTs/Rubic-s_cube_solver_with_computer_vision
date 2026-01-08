async function sendSolveRequest() {
        try {
            console.log("here")
            let object = {}
            object[0] = sharedCubeState.frontSide;
            object[1] = sharedCubeState.topSide;
            object[2] = sharedCubeState.leftSide;
            object[3] = sharedCubeState.rightSide;
            object[4] = sharedCubeState.bottomSide;
            object[5] = sharedCubeState.backSide;

            for(let i = 0; i < 6; i++) {
               object[i] = object[i].map((element, index) => {return sharedCubeState.number_to_color[element]})
            }

            const response = await fetch("http://localhost:8000/solve", {
                method: "POST",
                headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
                },
                body: JSON.stringify(object)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            console.log("Server response:", data);

            if (data.status == 'ok') {
                // console.log(`Image processed`);
                console.log(data.greet);

                // for(let i = 0; i < 9; i++) {
                //     // don't change center color
                //     if(i == 4) continue;
                //     side[i] = sharedCubeState.number_to_color[data.predictions[i]];
                // }
            } else {
                console.error("Processing failed:", data.message);
            }
        } catch (error) {
            window.alert("Upload failed:", error);
        }


    }