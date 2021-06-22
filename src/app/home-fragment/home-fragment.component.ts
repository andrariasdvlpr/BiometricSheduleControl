import { Component, OnInit } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

@Component({
  selector: 'app-home-fragment',
  templateUrl: './home-fragment.component.html',
  styleUrls: ['./home-fragment.component.css']
})
export class HomeFragmentComponent implements OnInit {

  showForm : boolean = false;
  suggestion! : string;

  constructor(public snackBar : MatSnackBar, public restService :RestService, public matDialog : MatDialog) { }

  ngOnInit(): void {
  }

  showFormSuggestions(){
    this.showForm = true;
  }

  sendSuggestion(){
    console.log(this.suggestion);
    if(this.suggestion){
      const dialog = this.openProgressDialog();
      const sugg ={ suggestion_text : this.suggestion};
      this.restService.doSuggestion(sugg).subscribe(
        _data => this.openSnackBar("Envio correcto, ¡Gracias!",false),
        _err => this.openSnackBar("Error en el envío", true),
        () => {
          this.showForm = false;
          this.suggestion = "";
          dialog.close();
        }
      );
    }else{
      this.openSnackBar("Complete el campo para enviar",true);
    }
  }

  openSnackBar(message : string, isError :boolean) {
    var styleClass = "success-snackbar";
    if(isError){
      styleClass ="error-snackbar";
    }
    this.snackBar.open(message,'', {duration: 5000 ,panelClass: styleClass});
  }

  openProgressDialog() : MatDialogRef<any>{
    const dialogConfig = new MatDialogConfig();
      // The user can't close the dialog by clicking outside its body
      dialogConfig.id = "map-modal-component";
      dialogConfig.maxHeight ="90%";
      dialogConfig.disableClose = true;
      const modalDialog= this.matDialog.open(ProgressDialogComponent, dialogConfig);
      return modalDialog;
  }
}
