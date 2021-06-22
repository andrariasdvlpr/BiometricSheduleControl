import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PasswordFragmentComponent } from './password-fragment.component';

describe('PasswordFragmentComponent', () => {
  let component: PasswordFragmentComponent;
  let fixture: ComponentFixture<PasswordFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PasswordFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(PasswordFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
