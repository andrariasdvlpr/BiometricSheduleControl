import { Component, OnInit } from '@angular/core';
import { MatDialog, MatDialogConfig, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ProgressDialogComponent } from '../progress-dialog/progress-dialog.component';
import { RestService } from '../rest.service';

interface Type {
  value: string;
  viewValue: string;
}

@Component({
  selector: 'app-camera-fragment',
  templateUrl: './camera-fragment.component.html',
  styleUrls: ['./camera-fragment.component.css']
})
export class CameraFragmentComponent implements OnInit {

  value : any;
  isShowResult : boolean = false;
  initialState : boolean = true;
  dialog!: MatDialogRef<any, any>;
  selectedValue! : string;

  types: Type[] = [
    {value: '1', viewValue: 'Inicio jornada laboral'},
    {value: '2', viewValue: 'Finalización jornada laboral'},
    {value: '3', viewValue: 'Salida a descanso'},
    {value: '4', viewValue: 'Entrada de descanso'},
    {value: '5', viewValue: 'Inicio horario de comer'},
    {value: '6', viewValue: 'Finalización horario de comer'}
  ];

  constructor(public restService : RestService, public matDialog : MatDialog, public snackBar : MatSnackBar) { }

  ngOnInit(): void {
  }

  showResult(data : any){
    this.dialog = this.openProgressDialog();
    this.restService.doFacialVerification(data,this.selectedValue).subscribe(
      data => {
        this.shapeResponse(data)
      },
      err => {
        this.dialog.close()
        this.openSnackBar(err.statusText, "error-snackbar");
      },
      () =>  this.dialog.close()
    );
    
  }

  shapeResponse(data : any){
    this.value = data
    this.isShowResult = true;
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

  openSnackBar(message : string, classStyle : string) {
    this.snackBar.open(message,'', {duration: 5000 ,panelClass: classStyle});
  }

  onOptionsSelected(){
    this.initialState = false;
  }
}
