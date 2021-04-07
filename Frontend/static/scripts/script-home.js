function changeSlide(num)
{
    current_slide += num;
    showSlide(current_slide);
}

function goToSlide(num)
{
    current_slide = num;
    showSlide(current_slide);
}

function showSlide(num) 
{
    var slides = document.getElementsByClassName("slide");
    var dots = document.getElementsByClassName("dot");

    if(num > slides.length)
    {
        current_slide = 1;
    }

    if(num < 1)
    {
        current_slide = slides.length;
    }

    var i;
    for(i=0; i<slides.length; i++)
    {
        slides[i].style.display = "none";
    }

    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }

    slides[current_slide-1].style.display = "block";
    dots[current_slide-1].className += " active"
}


var current_slide = 1;
showSlide(current_slide);
setInterval(function(){
    changeSlide(1);
}, 3000);