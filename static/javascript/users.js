function deleteSubmit() {
    submitAction = document.getElementById("manageForm");
    submitAction.action ="delete";
    submitAction.submit();
}
function getuserdata(userid){
    var xmlhttp = new XMLHttpRequest();
    
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4) {
            if(this.status == 200){

                var user = JSON.parse(this.responseText);  
                document.getElementById("employee_id_hidden").value = user.employee_id;
                document.getElementById("employee_id").value = user.employee_id;
                document.getElementById("firstname").value = user.first_name;
                document.getElementById("lastname").value = user.last_name;
                document.getElementById("username").value = user.username;
                document.getElementById("email").value = user.email;
                if(user.image.localeCompare("null")!=0){
                    document.getElementById('imgPopUp').setAttribute('src', 'data:image/png;base64,'+user.image);
                }
                else{
                    document.getElementById('imgPopUp').setAttribute('src', '/static/styleResources/account-circle.png');
                }
                document.getElementById('registrationDate').value = user.registration_date;
                document.getElementById('threshold').value= user.decision_threshold;
                document.getElementById('modal-wrapper').style.display='flex';


            }
            else if(this.status == 404){
            
            }

        }
    };
    var url = "http://192.168.0.16/bsc/rest-user/"+userid;
    xmlhttp.open("GET", url, true);
    xmlhttp.setRequestHeader("Authorization", "Token "+ getCookie("FC_SESSIONID") )
    xmlhttp.send();
}

function getCookie(cname) {
    var name = cname + "=";
    var decodedCookie = decodeURIComponent(document.cookie);
    var ca = decodedCookie.split(';');
    for(var i = 0; i <ca.length; i++) {
      var c = ca[i];
      while (c.charAt(0) == ' ') {
        c = c.substring(1);
      }
      if (c.indexOf(name) == 0) {
        return c.substring(name.length, c.length);
      }
    }
    return "";
  }