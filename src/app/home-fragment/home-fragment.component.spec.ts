import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HomeFragmentComponent } from './home-fragment.component';

describe('HomeFragmentComponent', () => {
  let component: HomeFragmentComponent;
  let fixture: ComponentFixture<HomeFragmentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HomeFragmentComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(HomeFragmentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
