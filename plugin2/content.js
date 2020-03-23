var sentences = [];

setInterval(function(){
  sentences = [];
}, 10000);

setInterval(function () {
  var spans = document.getElementsByClassName("_3l3x");
  for (let span of spans) {
    var currentSentence = span.innerText.trim();

    if (!sentences.includes(currentSentence)) {
      console.log("new sentence: " + currentSentence);

      $.ajax({
        url: 'http://127.0.0.1:8000/api/',
        dataType: 'json',
        type: 'post',
        contentType: 'application/json',
        data: JSON.stringify({ sentence: currentSentence }),
        processData: false,
        success: function (data, textStatus, jQxhr) {
          let serverResponse = JSON.stringify(data);
          console.log(data);

          if (serverResponse.includes('0')) {
            span.style["background-color"] = "yellow";
            // span.style["border"] = "thick solid #FF0000";
               // span.style.color = 'red';
          }

         
        },
        error: function (jqXhr, textStatus, errorThrown) {
          console.log(errorThrown);
        }
      });
      sentences.push(currentSentence);
      console.log("Total sentences: "+sentences.length);
    }
  }

  console.log("console Go");
}, 500);
