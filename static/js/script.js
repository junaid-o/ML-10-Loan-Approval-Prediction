// Thanks to Pavel Dobryakov //

const ANGLE = 40;

let card = document.querySelectorAll(".card");

card.forEach((element, i) => {
  floatable(element);
});

function floatable(panel) {
  let content = panel.querySelector(".content");
  panel.addEventListener('mouseout', e => {
    content.style.transform = `perspective(400px)
                   rotateX(0deg)
                   rotateY(0deg)
                   rotateZ(0deg)
                    translateZ(40px)`;
    content.style.transition = `all 2s linear`;
  });

  panel.addEventListener('mousemove', e => {
    let w = panel.clientWidth;
    let h = panel.clientHeight;
    let y = (e.offsetX - w * 0.5) / w * ANGLE;
    let x = (1 - (e.offsetY - h * 0.5)) / h * ANGLE;

    content.style.transform = `perspective(400px)
                   rotateX(${x}deg)
                   rotateY(${y}deg)`;
  });
}